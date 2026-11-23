# Pilot rà soát nhãn Phase 11

Bộ 40 mẫu private test dùng để phân tích lỗi và thử cách đánh giá đóng góp của ảnh. Mẫu được chọn có chủ đích, nên tỷ lệ thu được chỉ mô tả bộ pilot này.

## Chuẩn bị

Người điều phối đọc `coordinator_key.csv` để lấy ảnh gốc từ `private-test-images/test-images/`. Cột `source_image` là tên file gốc và `review_image` là đường dẫn đích tương đối với thư mục này. Cần chép đủ 40 ảnh theo ánh xạ trước khi mở lượt ảnh. Workspace hiện chỉ có metadata ảnh. Giữ bảng ánh xạ cho người điều phối vì bảng có nhãn benchmark và nhóm chọn mẫu.

Đối chiếu metadata đã xác nhận caption và nhãn benchmark khớp predictions ở cả 1.504 ID. Bộ pilot gồm 15 mẫu `image` có đổi đúng/sai, toàn bộ 7 mẫu `text`, 12 mẫu `not` mất dự đoán đúng ở nhiều seed và 6 mẫu đối chiếu có trạng thái đúng/sai giống nhau giữa CE và LA trong từng seed. Trong 12 mẫu `not`, có 4 mẫu mất ở cả ba seed và 8 mẫu còn lại chuyển sang `multi` ở ít nhất hai seed. Các mẫu đối chiếu có 2 mẫu mỗi lớp `not`, `image`, `multi`. Các lượt lấy ngẫu nhiên và xáo thứ tự dùng seed 42.

## Hai người đánh giá độc lập

Mỗi người dùng bản sao riêng. Phát lần lượt từng phiếu và thu phiếu trước khi phát lượt tiếp theo. Giữ kín nhãn benchmark, dự đoán và kết quả của người còn lại đến khi nộp đủ ba lượt.

1. `01_caption.csv`. Chỉ xem caption. Quyết định caption đứng riêng có châm biếm hay không.
2. `02_image.csv`. Chỉ xem ảnh, kể cả chữ hiển thị trong ảnh. Quyết định ảnh đứng riêng có châm biếm hay không.
3. `03_pair.csv`. Xem caption và ảnh cùng nhau. Quyết định cả bài có châm biếm hay không và ghi vai trò của ảnh.

Điền `sarcasm` bằng `co`, `khong` hoặc `khong_chac`. Điền `confidence` bằng 1 (thấp), 2 (vừa) hoặc 3 (cao). Trong `evidence`, ghi cụm từ, chi tiết thị giác hoặc sự tương phản làm căn cứ. Nếu cần bối cảnh bên ngoài, ghi bối cảnh còn thiếu. Nếu ảnh không mở được, ghi `thieu_anh` trong `evidence` và để trống các ô đánh giá phụ thuộc vào ảnh.

Ở lượt cả cặp, điền `image_contribution` bằng `can` nếu bỏ ảnh thì mất bằng chứng cần để nhận ra ý châm biếm, `bo_sung` nếu caption đã đủ nhưng ảnh làm rõ thêm, `khong` nếu ảnh không đóng góp, hoặc `khong_chac` nếu chưa xác định. Có thể dùng hướng dẫn gán nhãn của repo để thống nhất khái niệm châm biếm trước khi bắt đầu.

## Tổng hợp

Ghép phiếu bằng `audit_id` sau khi cả hai hoàn tất. Báo tỷ lệ đồng thuận thô cho từng lượt và cho vai trò của ảnh, số trường hợp thiếu ảnh hoặc không chắc, cùng ví dụ bất đồng. Giữ cả hai đánh giá gốc và ghi quyết định thống nhất vào bảng riêng. Đánh dấu mẫu `cần ảnh` khi đánh giá đã thống nhất cho thấy caption đứng riêng không đủ, cả cặp có châm biếm và ảnh cung cấp bằng chứng cần thiết. Sau đó mở nhãn benchmark để phân tích mức độ khớp với đánh giá của hai người.

## Lượt gán nhãn bổ sung bằng AI

Hai AI rater gán nhãn trên cùng ba phiếu: `01_caption_ai_rater_{1,2}.csv`, `02_image_ai_rater_{1,2}.csv` và `03_pair_ai_rater_{1,2}.csv` (commit `45eb80f`). Các file dùng cùng cột với phiếu của người đánh giá. Kết quả AI được báo riêng và không gộp vào số đồng thuận giữa hai người (κ, tỷ lệ đồng thuận thô). Phân tích: `data/server/outputs/p11_controlled/analyze_label_audit.py` → `label_audit_analysis.json` và `image_needed_model_comparison.csv`; tóm tắt ở `reports/results/phase11_la_controlled.md` mục 8.
