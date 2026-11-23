# Rà soát nhãn vòng 2 (WS-C)

120 mẫu private test, không trùng 40 mẫu pilot (`../phase11_label_audit/`). Rút bằng
`python scripts/sample_audit_round2.py --seed 42 --write --copy-images` ngày 2026-11-23. Thiết kế và phân tích
được cố định trước khi gán nhãn trong `reports/plan/ws_c_round2.md`. Hướng dẫn v2 đã được nhóm duyệt ngày 2026-11-23.

## Dành cho người điều phối

- `coordinator_key.csv` chứa nhãn benchmark, nhóm chọn mẫu, trọng số và tín hiệu mô hình. **Không đưa file này cho
  người đánh giá.** Chỉ phát các file trong `batch1/`, `batch2/` và thư mục `images/`.
- `images/` đã có đủ 120 ảnh (`R###.jpg`), chép từ `private-test-images/test-images/` theo `source_image`.
- Làm **batch 1 trước**, đủ ba lượt cho cả hai người, hạn 1/12. Sau đó mới làm batch 2, hạn 7/12. Không giải thích
  hai batch khác nhau thế nào.
- Trong mỗi batch, phát lần lượt từng phiếu và thu phiếu trước khi phát lượt tiếp theo: `01_caption.csv` →
  `02_image.csv` → `03_pair.csv`. Mỗi người dùng bản sao riêng, đặt tên `<phiếu>_person_1.csv` /
  `<phiếu>_person_2.csv` trong đúng thư mục batch.
- Không dùng AI để nháp hay gợi ý. Giữ kín nhãn benchmark, dự đoán mô hình và kết quả của người còn lại đến khi nộp đủ
  ba lượt của cả hai batch.

## Hướng dẫn cho người đánh giá

1. `01_caption.csv`: chỉ xem caption. Quyết định caption đứng riêng có châm biếm hay không.
2. `02_image.csv`: chỉ xem ảnh, kể cả chữ hiển thị trong ảnh. Quyết định ảnh đứng riêng có châm biếm hay không.
3. `03_pair.csv`: xem caption và ảnh cùng nhau. Quyết định cả bài có châm biếm hay không, rồi ghi vai trò của ảnh.

Điền `sarcasm` bằng `co`, `khong` hoặc `khong_chac`. Điền `confidence` bằng 1 (thấp), 2 (vừa) hoặc 3 (cao). Trong
`evidence`, ghi cụm từ, chi tiết thị giác hoặc sự tương phản làm căn cứ. Nếu cần bối cảnh bên ngoài, ghi bối cảnh còn
thiếu. Nếu ảnh không mở được, ghi `thieu_anh` trong `evidence` và để trống các ô phụ thuộc vào ảnh.

### Vai trò của ảnh (`image_contribution`, chỉ ở lượt cả cặp). Điểm mới của v2

- **Chỉ điền khi `sarcasm` = `co`.** Nếu `sarcasm` là `khong` hoặc `khong_chac`, điền `NA`.
- Cách phân biệt: che ảnh và chỉ đọc caption.
  - Không còn nhận ra ý châm biếm → `can`.
  - Vẫn nhận ra, nhưng ảnh làm rõ thêm → `bo_sung`.
  - Vẫn nhận ra và ảnh không thêm gì → `khong`.
  - Chưa xác định được → `khong_chac`.

### Ví dụ (lấy từ pilot, không có trong vòng 2)

| Mẫu pilot | Cả cặp | Vai trò ảnh | Vì sao |
|---|---|---|---|
| A008 | `co` | `can` | Caption chỉ là một câu hỏi bình thường; câu trả lời trong ảnh mới tạo ra ý mỉa mai. Che ảnh thì mất châm biếm. |
| A033 | `co` | `khong` | Caption đứng riêng đã châm biếm; ảnh không thêm bằng chứng. |
| A002 | `khong` | `NA` | Tin chia sẻ thông tin về hồ nước Hà Nội. Bài không châm biếm nên không đánh giá vai trò ảnh, dù ảnh có thêm thông tin. |

(Ảnh và caption của các mẫu này nằm trong `../phase11_label_audit/`.)

## Tổng hợp (sau khi nộp)

Phân tích theo `reports/plan/ws_c_round2.md` §5:
- Gate A chỉ dùng batch 1;
- độ tin cậy được tính trên batch 1, có trọng số;
- batch 2 chỉ dùng để mô tả.

Pilot và vòng 2 được báo cáo riêng.
