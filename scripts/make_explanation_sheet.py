"""Sheet Excel để người viết giải thích cho mẫu train (thay teacher LLM, xem src/vimmsd/data/human_rationale.py).

    python scripts/make_explanation_sheet.py

Ghi vào <output_dir>/human_explanations/: explanations_v1.xlsx (ảnh thu nhỏ trong từng dòng), images/ (ảnh gốc, tên
<STT>_<id>.jpg để phóng to) và ids.txt. Không ghi đè sheet đã có: người viết có thể đã điền vào đó."""
import argparse
import io
import random
import shutil
from pathlib import Path

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from vimmsd.data.human_rationale import DEFAULT_QUOTA, collect_drafts, interleave_batches, sample_ids
from vimmsd.data.image_io import open_image_rgb
from vimmsd.data.dataset import rationale_dir
from vimmsd.data.rationale import read_ids, split_records, write_ids
from vimmsd.utils.config import load_config

# thứ tự ưu tiên nháp: prompt mới trước, Pro trước Flash Lite cùng phiên bản
DRAFT_DIRS = ("v8_natural_dev_gemini-3.5-flash-lite", "v7_natural_dev_gemini-3.5-flash-lite", "v5_pilot_dev_gemini",
              "v6_pilot_dev_gemini-3.5-flash-lite", "v6_natural_dev_gemini-3.5-flash-lite")
YES_NO = '"Có,Không"'
THUMB = 300  # px, cạnh dài của ảnh trong sheet

INSTRUCTIONS = """HƯỚNG DẪN VIẾT GIẢI THÍCH

Mục đích: giải thích vì sao bài đăng có nhãn đã cho, để dạy model nhỏ cách suy luận. Nhãn đã có sẵn, KHÔNG gán lại nhãn.

Các bước cho mỗi dòng (xem định nghĩa trong docs/labeling_guidelines.md):
1. Xem ảnh (bấm vào thư mục images/, tên file bắt đầu bằng STT để phóng to) và đọc caption, chữ trong ảnh.
2. Cột "Caption riêng có mỉa mai?": chỉ đọc caption, che ảnh đi. Có nói ngược, phóng đại, châm biếm không?
3. Cột "Ảnh riêng có mỉa mai?": chỉ xem ảnh (kể cả chữ trong ảnh), bỏ qua caption.
4. Cột "Ghép lại tạo mỉa mai mới?": caption và ảnh mâu thuẫn nhau, hoặc ghép lại mới thấy ý mỉa mai.
5. Viết 3 lý do, mỗi lý do 1–2 câu tiếng Việt, nêu bằng chứng cụ thể (trích chữ trong caption, chi tiết trong ảnh).
6. Cột "Kiểm tra" báo nếu 3 cột Có/Không không khớp với nhãn theo bảng quyết định. Sửa lại, hoặc nếu bạn thật sự
   thấy nhãn sai thì đánh dấu "Không đồng ý với nhãn" và ghi lý do vào Ghi chú, không cần viết 3 lý do.

Bảng quyết định:
- Không / Không / Không -> not-sarcasm
- Có / Không / Không -> text-sarcasm
- Không / Có / Không -> image-sarcasm
- Cả hai Có, hoặc "Ghép lại" Có -> multi-sarcasm

Quy tắc khi viết lý do:
- Không viết tên nhãn (vd "đây là multi-sarcasm"). Lý do phải nói vì sao, không nhắc lại kết luận.
- Không đoán. Không hiểu meme thì ghi "không hiểu" vào Ghi chú và bỏ qua dòng đó.
- Dòng có cột "Nháp" là lý do do AI viết sẵn (chỉ khi AI tự đoán đúng nhãn). Đọc kỹ, sửa chỗ sai, xoá chỗ bịa.
  Đừng giữ nguyên chỉ vì nó nghe hợp lý.
- Ghi tên vào cột Người viết và số phút đã dùng, để ước tính thời gian cho các đợt sau.

Đợt: sheet chia 4 đợt, mỗi đợt khoảng 100 mẫu cùng tỉ lệ lớp. Làm xong đợt 1 thì cả nhóm đọc lại một số dòng, thống
nhất thế nào là lý do tốt, rồi mới làm tiếp.

Sheet "Doi_chieu": 50 mẫu cũng có trong sheet chính nhưng không có nháp. Người THỨ HAI điền độc lập, không xem
bản của người kia, để đo mức đồng thuận."""

COLUMNS = [("STT", 6), ("Đợt", 5), ("ID", 8), ("Nhãn", 14), ("Ảnh", 44), ("Caption", 50), ("Chữ trong ảnh (OCR)", 40),
           ("Nháp", 10), ("Caption riêng có mỉa mai?", 12), ("Ảnh riêng có mỉa mai?", 12), ("Ghép lại tạo mỉa mai mới?", 12),
           ("Lý do: caption", 45), ("Lý do: ảnh", 45), ("Lý do: khi ghép", 45), ("Kiểm tra", 18),
           ("Không đồng ý với nhãn", 12), ("Ghi chú", 30), ("Người viết", 12), ("Phút", 6)]


def thumbnail(path):
    img = open_image_rgb(path)
    img.thumbnail((THUMB, THUMB))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    buf.seek(0)
    return XLImage(buf), img.size


def check_formula(row):
    """Nhãn suy từ 3 cột Có/Không (I, J, K) theo bảng quyết định, so với cột Nhãn (D)."""
    i, j, k = f"I{row}", f"J{row}", f"K{row}"
    derived = (f'IF(OR(AND({i}="Có",{j}="Có"),{k}="Có"),"multi-sarcasm",IF({i}="Có","text-sarcasm",'
               f'IF({j}="Có","image-sarcasm","not-sarcasm")))')
    return f'=IF(OR({i}="",{j}="",{k}=""),"",IF({derived}=D{row},"khớp","Mâu thuẫn với nhãn"))'


def write_sheet(ws, rows, records, drafts):
    bold = Font(bold=True)
    for c, (name, width) in enumerate(COLUMNS, start=1):
        cell = ws.cell(1, c, name)
        cell.font = bold
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[cell.column_letter].width = width
    ws.freeze_panes = "E2"
    yes_no = DataValidation(type="list", formula1=YES_NO, allow_blank=True)
    agree = DataValidation(type="list", formula1='"Không đồng ý"', allow_blank=True)
    ws.add_data_validation(yes_no)
    ws.add_data_validation(agree)
    wrap = Alignment(wrap_text=True, vertical="top")
    for r, (stt, batch, sid) in enumerate(rows, start=2):
        rec = records[sid]
        draft = drafts.get(sid)
        values = [stt, batch, sid, rec["label_name"], None, rec["caption"], rec.get("ocr", ""), "có" if draft else ""]
        values += [None] * 3
        values += [(draft[1].get(k) or "") if draft else "" for k in ("text_reason", "image_reason", "combined_reason")]
        values += [check_formula(r), None, None, None, None]
        for c, v in enumerate(values, start=1):
            ws.cell(r, c, v).alignment = wrap
        img, (w, h) = thumbnail(rec["image_path"])
        ws.add_image(img, f"E{r}")
        ws.row_dimensions[r].height = max(h, 60) * 0.75 + 6  # điểm = px * 0,75
        yes_no.add(f"I{r}:K{r}")
        agree.add(f"P{r}")
    red = PatternFill(start_color="F4CCCC", end_color="F4CCCC", fill_type="solid")
    ws.conditional_formatting.add(f"O2:O{len(rows) + 1}", FormulaRule(formula=['O2="Mâu thuẫn với nhãn"'], fill=red))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--agreement", type=int, default=50, help="số mẫu cho người thứ hai viết độc lập")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config("configs/base.yaml")
    records = split_records(cfg, "train")
    root = rationale_dir(cfg)
    exclude = read_ids(root / "ids" / "pilot_eval.txt")  # để dành cho đánh giá teacher, không dùng ở đây
    chosen = sample_ids({sid: r["label_name"] for sid, r in records.items()}, DEFAULT_QUOTA, exclude, args.seed)
    order = interleave_batches(chosen, args.batches, args.seed)
    drafts = collect_drafts([root / "raw" / d for d in DRAFT_DIRS if (root / "raw" / d).exists()])
    wanted = {sid for _, sid in order}
    drafts = {sid: d for sid, d in drafts.items() if sid in wanted}
    rows = [(stt, batch, sid) for stt, (batch, sid) in enumerate(order, start=1)]

    out = Path(cfg.paths.output_dir) / "human_explanations"
    sheet = out / "explanations_v1.xlsx"
    if sheet.exists():
        raise SystemExit(f"{sheet} đã có, có thể đã được điền. Xoá hoặc đổi tên file nếu thật sự muốn tạo lại.")
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    names = {}
    for stt, _, sid in rows:
        names[sid] = f"{stt:03d}_{sid}.jpg"
        open_image_rgb(records[sid]["image_path"]).save(images / names[sid], "JPEG", quality=92)

    # mẫu đối chiếu: chỉ lấy dòng không có nháp, để cả hai người cùng viết từ đầu
    pool = [row for row in rows if row[2] not in drafts]
    agreement = sorted(random.Random(args.seed).sample(pool, min(args.agreement, len(pool))))

    wb = Workbook()
    guide = wb.active
    guide.title = "Huong_dan"
    for line_no, line in enumerate(INSTRUCTIONS.splitlines(), start=1):
        guide.cell(line_no, 1, line)
    guide.column_dimensions["A"].width = 120
    write_sheet(wb.create_sheet("Mau"), rows, records, drafts)
    write_sheet(wb.create_sheet("Doi_chieu"), agreement, records, {})
    wb.save(sheet)
    write_ids(out / "ids.txt", [sid for _, _, sid in rows])
    shutil.make_archive(str(out / "images"), "zip", images)
    counts = {label: len(ids) for label, ids in chosen.items()}
    print(f"đã ghi {sheet} ({len(rows)} mẫu {counts}, {len(drafts)} có nháp, {len(agreement)} mẫu đối chiếu)")
    print(f"ảnh gốc: {images} và {out / 'images.zip'}")


if __name__ == "__main__":
    main()
