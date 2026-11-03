"""Teacher chạy qua Gemini REST, cùng định dạng output với generate_rationales.py (eval_teacher.py đọc được).

    python scripts/generate_rationales_gemini.py --model gemini-3.5-flash-lite --split train \\
        --ids-file <rationales>/ids/natural_dev.txt --out-dir <rationales>/raw/v8_natural_dev_gemini-3.5-flash-lite

Key đọc từ biến môi trường GEMINI_API_KEY hoặc file .env (không đưa vào git). Chạy tiếp được: mẫu đã có output thì bỏ
qua. Chạy này tốn tiền: số token đầu vào và đầu ra in ở cuối. --limit để thử trước vài mẫu."""
import argparse
import base64
import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vimmsd.data.image_io import open_image_rgb
from vimmsd.data.rationale import DATASET_EXAMPLE_IDS, PROMPT_VERSION, build_prompt, parse_teacher_output, read_ids, split_records
from vimmsd.utils.config import load_config

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRY_CODES = (429, 500, 502, 503, 504)
SAFETY = [{"category": c, "threshold": "BLOCK_NONE"} for c in ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                                                               "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")]


def load_api_key(env_path=".env"):
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    path = Path(env_path)
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        m = re.match(r"\s*(?:export\s+)?GEMINI_API_KEY\s*=\s*(.*)", line)
        if m:
            return m.group(1).strip().strip("'\"")
    raise SystemExit("không thấy GEMINI_API_KEY trong môi trường hoặc .env")


def request_body(record, temperature):
    buf = io.BytesIO()
    open_image_rgb(record["image_path"]).save(buf, "JPEG", quality=90)
    parts = [{"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(buf.getvalue()).decode()}},
             {"text": build_prompt(record["caption"], record.get("ocr", ""))}]
    return json.dumps({"contents": [{"role": "user", "parts": parts}], "generationConfig": {"temperature": temperature},
                       "safetySettings": SAFETY}).encode()


def call_api(url, body, key, attempts=8):
    """Trả về (JSON, None) hoặc (None, lỗi). Không ghi key vào thông báo lỗi."""
    error = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, data=body, headers={"x-goog-api-key": key, "Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response), None
        except urllib.error.HTTPError as e:
            error = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
            if e.code not in RETRY_CODES:
                break
        except OSError as e:
            error = type(e).__name__
        time.sleep(min(120, 5 * 2 ** attempt))
    return None, error


def to_row(sid, split, record, response, model, temperature, seconds):
    candidate = (response.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in candidate.get("content", {}).get("parts", []) if not p.get("thought"))
    return {"id": sid, "split": split, "sample_idx": 0, "label": record["label_name"], "with_label": False, "output": text,
            "parsed": parse_teacher_output(text), "teacher": model, "model_version": response.get("modelVersion"),
            "finish_reason": candidate.get("finishReason"), "block": response.get("promptFeedback", {}).get("blockReason"),
            "usage": response.get("usageMetadata", {}), "prompt_version": PROMPT_VERSION, "temperature": temperature,
            "seconds": round(seconds, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ids-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "public_test", "private_test"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    key = load_api_key()
    records = split_records(load_config("configs/base.yaml"), args.split)
    ids = [i for i in read_ids(args.ids_file) if i not in DATASET_EXAMPLE_IDS or args.split != "train"][:args.limit]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.split}.shard0of1.jsonl"
    done = {json.loads(line)["id"] for line in path.open(encoding="utf-8")} if path.exists() else set()
    todo = [i for i in ids if i not in done]
    print(f"cần chạy {len(todo)}, đã có {len(done)}", flush=True)

    url = API_URL.format(model=args.model)
    lock = threading.Lock()
    totals = {"prompt": 0, "output": 0, "thoughts": 0, "failed": 0, "empty": 0}

    def work(sid):
        start = time.time()
        response, error = call_api(url, request_body(records[sid], args.temperature), key)
        if response is None:
            with lock:
                totals["failed"] += 1
                print(f"lỗi {sid}: {error}", flush=True)
            return
        row = to_row(sid, args.split, records[sid], response, args.model, args.temperature, time.time() - start)
        with lock:
            totals["prompt"] += row["usage"].get("promptTokenCount", 0)
            totals["output"] += row["usage"].get("candidatesTokenCount", 0)
            totals["thoughts"] += row["usage"].get("thoughtsTokenCount", 0)
            totals["empty"] += not row["output"]
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, todo))
    print(f"tổng {totals}", flush=True)


if __name__ == "__main__":
    main()
