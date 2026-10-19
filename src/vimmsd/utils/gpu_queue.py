"""Chạy nhiều run độc lập song song, mỗi GPU một run (Kaggle T4 x2).

Model Phase 4 chỉ dùng khoảng 1-1.5 GB VRAM nên DDP cho một run không đáng: chia batch 16 cho 2 GPU còn tốn
đồng bộ gradient. Chạy 2 run cùng lúc thì tổng thời gian giảm gần một nửa mà không phải sửa trainer.
Mỗi GPU có một luồng lấy job từ hàng đợi chung, chạy lần lượt các lệnh của job với CUDA_VISIBLE_DEVICES=k
(trong tiến trình con GPU đó là cuda:0), log ghi ra file của job."""
import os
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Job:
    name: str
    commands: list            # mỗi lệnh là list[str]; lệnh sau chỉ chạy khi lệnh trước thành công
    log: Path
    check: object = None      # hàm () -> str | None, chạy sau khi xong mọi lệnh; trả về chuỗi lỗi nếu job hỏng
                              # (có thể dọn file tạm ở đây, vd xoá last.pt khi run đã xong)
    returncode: int = field(default=None, init=False)
    error: str = field(default=None, init=False)
    minutes: float = field(default=None, init=False)


def run_on_gpus(jobs, num_gpus, summary_pattern=None, tail_on_error=30):
    """Chạy `jobs` trên `num_gpus` GPU, trả về danh sách job đã chạy (kèm returncode, error, minutes).

    num_gpus = 0 (máy không có GPU) thì chạy tuần tự, không đặt CUDA_VISIBLE_DEVICES.
    `summary_pattern`: regex, job xong thì in các dòng log khớp (vd các dòng `INFO epoch`).
    Ngắt kernel (KeyboardInterrupt) thì dừng luôn các tiến trình con đang chạy."""
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    lock = threading.Lock()
    running = {}
    stop = threading.Event()
    pattern = re.compile(summary_pattern) if summary_pattern else None

    def say(msg):
        with lock:
            print(msg, flush=True)

    def worker(gpu):
        env = {**os.environ, "TQDM_DISABLE": "1", "PYTHONUNBUFFERED": "1"}
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        tag = f"[gpu{gpu}]" if gpu is not None else "[cpu]"
        while not stop.is_set():
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            say(f"{tag} bắt đầu {job.name} (log: {job.log})")
            start = time.time()
            try:
                job.log.parent.mkdir(parents=True, exist_ok=True)
                with open(job.log, "w", encoding="utf-8") as log:
                    for cmd in job.commands:
                        log.write(f"$ {' '.join(map(str, cmd))}\n")
                        log.flush()
                        proc = subprocess.Popen([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT, env=env)
                        with lock:
                            running[gpu] = proc
                        job.returncode = proc.wait()
                        with lock:
                            running.pop(gpu, None)
                        if job.returncode != 0 or stop.is_set():
                            break
                if job.returncode != 0:
                    job.error = f"mã thoát {job.returncode}"
                elif job.check is not None:
                    job.error = job.check()
            except Exception as e:  # vd hết dung lượng đĩa khi tạo thư mục/log: báo lỗi job, luồng vẫn chạy job tiếp theo
                job.returncode = -1 if job.returncode is None else job.returncode
                job.error = f"{type(e).__name__}: {e}"
            job.minutes = (time.time() - start) / 60
            try:
                text = job.log.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                text = []
            if job.error:
                say(f"{tag} LỖI {job.name}: {job.error}, {tail_on_error} dòng log cuối:\n" + "\n".join(text[-tail_on_error:]))
            else:
                lines = [l for l in text if pattern.search(l)] if pattern else []
                say(f"{tag} xong {job.name} ({job.minutes:.1f} phút)" + "".join(f"\n    {l}" for l in lines))

    gpus = list(range(num_gpus)) if num_gpus > 0 else [None]
    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=1)
    except KeyboardInterrupt:
        stop.set()
        with lock:
            for proc in running.values():
                proc.terminate()
        raise
    return [job for job in jobs if job.returncode is not None]
