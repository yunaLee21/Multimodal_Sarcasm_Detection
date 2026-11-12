"""Augmentation caption lớp hiếm có duyệt tay (docs/empirical_analysis_plan.md mục 8.2). Ba bước:

    VIMMSD_ENV=server python scripts/augment_minority.py generate --config configs/base.yaml
    VIMMSD_ENV=server python scripts/augment_minority.py sheet --reviewers A B
    VIMMSD_ENV=server python scripts/augment_minority.py import --reviews A=<dir>/review_A.csv B=<dir>/review_B.csv

generate: back-translation vi -> en -> vi (BackTranslator) cho mọi mẫu train thuộc --labels, tối đa --candidates ứng
viên mỗi mẫu gốc, seed cố định; loại tự động ứng viên rỗng, không đổi, trùng caption train. Ghi candidates.jsonl
(kèm nhóm và fold của mẫu gốc theo --splits-manifest). Đã có file cùng tham số thì không sinh lại.
sheet: mỗi người duyệt một file CSV riêng (thấy nhãn gốc và đường dẫn ảnh, không thấy kết quả model).
import: chỉ nhận ứng viên mọi người duyệt đều accept và đánh y mọi tiêu chí, giữ tối đa 1 mỗi mẫu gốc. Ghi
augmentation_manifest.jsonl (data.augmentation.manifest) và review_summary.json (đồng thuận, tỉ lệ nhận theo lớp, lý do
loại, thời gian duyệt), rồi tiền xử lý caption được nhận vào text cache. Thư mục mặc định:
data/processed/empirical/augmentation."""
import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from vimmsd.data import augment_review as A
from vimmsd.data.denoise import id_sort_key, normalize_caption
from vimmsd.utils.config import load_config
from vimmsd.utils.env import git_commit

OUT = "data/processed/empirical/augmentation"
VI2EN, EN2VI = "vinai/vinai-translate-vi2en-v2", "vinai/vinai-translate-en2vi-v2"


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def cmd_generate(args):
    import torch

    from vimmsd.data.augmentation import BackTranslator

    cfg = load_config(args.config, args.override)
    src = Path(cfg.paths.data_dir)
    train = json.loads((src / cfg.data.train_json).read_text(encoding="utf-8"))
    splits = json.loads(Path(args.splits_manifest).read_text(encoding="utf-8"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    settings = {"method": "back-translation vi->en->vi", "vi2en": VI2EN, "en2vi": EN2VI,
                "vi2en_generation": {"num_beams": 5, "max_length": 256},
                "en2vi_generation": {"do_sample": True, "top_p": 0.9, "max_length": 256},
                "candidates_per_parent": args.candidates, "seed": args.seed, "batch_size": args.batch_size,
                "labels": args.labels, "splits_manifest": args.splits_manifest}
    meta_path = out / "candidates_meta.json"
    if (out / "candidates.jsonl").exists() and meta_path.exists() and \
            json.loads(meta_path.read_text())["settings"] == settings:
        print(f"đã có {out / 'candidates.jsonl'} cùng tham số, không sinh lại")
        return
    parents = A.select_parents(train, args.labels)
    norm_train = {normalize_caption(r["caption"]) for r in train.values()}
    print(f"{len(parents)} mẫu gốc {dict(Counter(train[i]['label'] for i in parents))}")
    torch.manual_seed(args.seed)
    bt = BackTranslator(VI2EN, EN2VI, device="cuda" if torch.cuda.is_available() else "cpu")
    texts = bt([train[i]["caption"] for i in parents], batch_size=args.batch_size, num_return_sequences=args.candidates)
    rows = []
    for n, pid in enumerate(parents):
        seen = set()
        for rank in range(args.candidates):
            cand = texts[n * args.candidates + rank]
            reason = A.auto_screen(train[pid]["caption"], cand, norm_train, seen)
            if reason is None:
                seen.add(normalize_caption(cand))
            rows.append({"candidate_id": A.variant_id(pid, rank), "parent_id": pid, "rank": rank,
                         "label": train[pid]["label"], "image": train[pid]["image"],
                         "group": splits["groups"][pid], "fold": splits["folds"][pid],
                         "original_caption": train[pid]["caption"], "candidate_caption": cand, "auto_reject": reason})
    write_jsonl(out / "candidates.jsonl", rows)
    meta_path.write_text(json.dumps({"settings": settings, "git_commit": git_commit(short=False),
                                     "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                     "auto_reject": dict(Counter(r["auto_reject"] or "to_review" for r in rows))},
                                    indent=2, ensure_ascii=False))
    print(f"{len(rows)} ứng viên -> {out / 'candidates.jsonl'}; {dict(Counter(r['auto_reject'] or 'to_review' for r in rows))}")


SHEET_COLS = ["candidate_id", "parent_id", "label", "image_path", "original_caption", "candidate_caption",
              *A.CRITERIA, "verdict", "reason", "minutes"]


def cmd_sheet(args):
    cfg = load_config(args.config, args.override)
    image_dir = Path(cfg.paths.data_dir) / cfg.data.train_image_dir
    out = Path(args.out_dir)
    rows = [r for r in read_jsonl(out / "candidates.jsonl") if not r["auto_reject"]]
    order = {lab: i for i, lab in enumerate(args.label_order)}
    rows.sort(key=lambda r: (order.get(r["label"], 99), id_sort_key(r["parent_id"]), r["rank"]))
    for who in args.reviewers:
        path = out / f"review_{who}.csv"
        if path.exists() and not args.force:
            sys.exit(f"{path} đã có (có thể đang được duyệt); thêm --force để ghi đè")
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SHEET_COLS)
            w.writeheader()
            for r in rows:
                w.writerow({"candidate_id": r["candidate_id"], "parent_id": r["parent_id"], "label": r["label"],
                            "image_path": str(image_dir / r["image"]), "original_caption": r["original_caption"],
                            "candidate_caption": r["candidate_caption"]})
        print(f"{len(rows)} ứng viên -> {path}")


def cmd_import(args):
    out = Path(args.out_dir)
    cands = read_jsonl(out / "candidates.jsonl")
    to_review = [c for c in cands if not c["auto_reject"]]
    reviews = {}
    for spec in args.reviews:
        who, path = spec.split("=", 1)
        with open(path, encoding="utf-8-sig", newline="") as f:
            reviews[who] = {row["candidate_id"]: row for row in csv.DictReader(f)}
    merged = A.merge_reviews([c["candidate_id"] for c in to_review], reviews)
    chosen = A.choose_variants(to_review, merged)
    meta = json.loads((out / "candidates_meta.json").read_text())
    manifest = []
    for pid, c in sorted(chosen.items(), key=lambda x: id_sort_key(x[0])):
        m = merged[c["candidate_id"]]
        manifest.append({"variant_id": c["candidate_id"], "parent_id": pid, "group": c["group"], "fold": c["fold"],
                         "label": c["label"], "original_caption": c["original_caption"], "caption": c["candidate_caption"],
                         "method": meta["settings"]["method"], "models": [VI2EN, EN2VI],
                         "generation": meta["settings"], "generator_commit": meta["git_commit"],
                         "reviewer_verdicts": m["verdicts"]})
    write_jsonl(out / "augmentation_manifest.jsonl", manifest)

    names = list(reviews)
    kappa = (A.cohen_kappa([A.parse_verdict(reviews[names[0]].get(c["candidate_id"], {}).get("verdict")) for c in to_review],
                           [A.parse_verdict(reviews[names[1]].get(c["candidate_id"], {}).get("verdict")) for c in to_review])
             if len(names) >= 2 else None)
    parents_by_label = Counter(lab for lab in {c["parent_id"]: c["label"] for c in cands}.values())
    minutes = {who: sum(float(r.get("minutes") or 0) for r in rows.values()) for who, rows in reviews.items()}
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reviewers": names, "single_reviewer": len(names) < 2,
        "candidates": len(cands), "auto_rejected": dict(Counter(c["auto_reject"] for c in cands if c["auto_reject"])),
        "reviewed": len(to_review),
        "accepted_candidates": sum(m["accepted"] for m in merged.values()),
        "variants_kept": len(manifest),
        "acceptance_by_label": {lab: {"parents": n, "variants": sum(1 for v in manifest if v["label"] == lab),
                                      "rate": round(sum(1 for v in manifest if v["label"] == lab) / n, 3)}
                                for lab, n in parents_by_label.items()},
        "kappa_verdict": kappa,
        "rejection_reasons": dict(Counter(r for m in merged.values() if not m["accepted"] for r in m["reasons"])),
        "review_minutes": minutes,
    }
    (out / "review_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("reviewed", "accepted_candidates", "variants_kept", "acceptance_by_label",
                                              "kappa_verdict")}, ensure_ascii=False))
    if summary["single_reviewer"]:
        print("CẢNH BÁO: chỉ một người duyệt; kế hoạch yêu cầu hai người khi có thể (mục 8.2)")
    if manifest and not args.no_preprocess:
        from vimmsd.data.dataset import text_cache_dir
        from vimmsd.data.preprocessing import TextPreprocessor

        cfg = load_config(args.config, args.override)
        TextPreprocessor.from_config(cfg.data.text).process_all([v["caption"] for v in manifest],
                                                                cache_dir=Path(text_cache_dir(cfg)))
        print(f"đã tiền xử lý {len(manifest)} caption vào text cache")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "sheet", "import"):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/base.yaml")
        p.add_argument("--override", nargs="*", default=[])
        p.add_argument("--out-dir", default=OUT)
        if name == "generate":
            p.add_argument("--labels", nargs="+", default=["text-sarcasm", "image-sarcasm"])
            p.add_argument("--candidates", type=int, default=2)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--batch-size", type=int, default=16)
            p.add_argument("--splits-manifest", default="data/processed/empirical/splits/folds_k3_s42.json")
        if name == "sheet":
            p.add_argument("--reviewers", nargs="+", required=True)
            p.add_argument("--label-order", nargs="+", default=["text-sarcasm", "image-sarcasm"])
            p.add_argument("--force", action="store_true")
        if name == "import":
            p.add_argument("--reviews", nargs="+", required=True, help="tên=đường_dẫn_csv")
            p.add_argument("--no-preprocess", action="store_true")
    args = parser.parse_args()
    {"generate": cmd_generate, "sheet": cmd_sheet, "import": cmd_import}[args.cmd](args)


if __name__ == "__main__":
    main()
