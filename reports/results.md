# Kết quả thí nghiệm

Mỗi lần train xong, `scripts/train.py` in ra 1 dòng kết quả (và ghi vào `<output_dir>/<tên config>/results_row.md`). Chạy local thì dòng được tự thêm vào đây; chạy Kaggle/Colab thì copy dòng đó vào đây rồi commit.

| Ngày | Config | Commit | Môi trường | Val macro-F1 | Test macro-F1 | Thời gian train (phút) | Ghi chú |
|---|---|---|---|---|---|---|---|
| 2026-10-30 | `image_only` | `646475c` | kaggle | 0.3323 | 0.3422 | 8.5 | Task 5: CLIP Image-only \| peak VRAM 1.67 GB |
| 2026-10-30 | `image_only_vit` | `646475c` | kaggle | 0.3267 | 0.3313 | 10.5 | Task 6: ViT Image-only \| peak VRAM 2.33 GB |
| 2026-10-30 | `vlm_image_centric` (Vintern, không metadata) | `646475c` | kaggle | n/a | 0.0453 | ~7.0 | Task 7: zero-shot inference 1.504 mẫu \| invalid rate 0.0000 |
| 2026-10-30 | `vlm_image_centric` (Vintern, có OCR + mô tả) | `646475c` | kaggle | n/a | 0.0151 | ~7.3 | Task 7: zero-shot inference 1.504 mẫu \| invalid rate 0.0000 |
