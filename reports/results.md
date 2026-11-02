# Kết quả thí nghiệm

Mỗi lần train xong, `scripts/train.py` in ra 1 dòng kết quả (và ghi vào `<output_dir>/<tên config>/results_row.md`). Chạy local thì dòng được tự thêm vào đây; chạy Kaggle/Colab thì copy dòng đó vào đây rồi commit.

| Ngày | Config | Commit | Môi trường | Val macro-F1 | Test macro-F1 | Thời gian train (phút) | Ghi chú |
|---|---|---|---|---|---|---|---|
| 2026-10-30 | `image_only` | `646475c` | kaggle | 0.3323 | 0.3422 | 8.5 | Task 5: CLIP Image-only \| peak VRAM 1.67 GB |
| 2026-10-30 | `image_only_vit` | `646475c` | kaggle | 0.3267 | 0.3313 | 10.5 | Task 6: ViT Image-only \| peak VRAM 2.33 GB |
| 2026-10-30 | `vlm_image_centric` (Vintern, không metadata) | `646475c` | kaggle | n/a | 0.0453 | ~7.0 | Task 7: zero-shot inference 1.504 mẫu \| invalid rate 0.0000 |
| 2026-10-30 | `vlm_image_centric` (Vintern, có OCR + mô tả) | `646475c` | kaggle | n/a | 0.0151 | ~7.3 | Task 7: zero-shot inference 1.504 mẫu \| invalid rate 0.0000 |

## Phase 4 — Multimodal fusion (concat + MLP), 11 config × 3 seed

Toàn bộ 33 run chạy lại từ đầu trên `phuclkz/vimmsd-private-dataset` với `data.split=official` (code `1c8a749`): train 10.805, val = public test 1.413, **test = private test 1.504 mẫu** (support per-class: not 458 / text 7 / image 40 / multi 999), `split_seed=42` nên 3 seed dùng chung một tập test. Sẵn sàng so sánh trực tiếp giữa các config.

| 2026-11-02 | `text_only_frozen_s42` | `1c8a749` | kaggle | 0.3759 | 0.3712 | 3.9 | Phase 4 text_only_frozen seed 42 \| peak VRAM 0.89 GB |
| 2026-11-02 | `image_only_frozen_s42` | `1c8a749` | kaggle | 0.3481 | 0.3576 | 11.9 | Phase 4 image_only_frozen seed 42 \| peak VRAM 0.48 GB |
| 2026-11-02 | `fusion_concat_frozen_s42` | `1c8a749` | kaggle | 0.3665 | 0.3531 | 13.0 | Phase 4 fusion_concat_frozen seed 42 \| peak VRAM 1.26 GB |
| 2026-11-02 | `fusion_concat_ftlast_s42` | `1c8a749` | kaggle | 0.3781 | 0.3739 | 34.4 | Phase 4 fusion_concat_ftlast seed 42 \| peak VRAM 1.51 GB |
| 2026-11-02 | `text_only_frozen_cw_s42` | `1c8a749` | kaggle | 0.3739 | 0.3894 | 2.6 | Phase 4 text_only_frozen_cw seed 42 \| peak VRAM 0.89 GB |
| 2026-11-02 | `fusion_concat_frozen_reg_s42` | `1c8a749` | kaggle | 0.3959 | 0.3735 | 20.3 | Phase 4 fusion_concat_frozen_reg seed 42 \| peak VRAM 1.26 GB |
| 2026-11-02 | `fusion_concat_frozen_reg_cw_s42` | `1c8a749` | kaggle | 0.4065 | 0.4163 | 17.6 | Phase 4 fusion_concat_frozen_reg_cw seed 42 \| peak VRAM 1.26 GB |
| 2026-11-02 | `fusion_multistream_frozen_s42` | `1c8a749` | kaggle | 0.3684 | 0.3722 | 10.0 | Phase 4 fusion_multistream_frozen seed 42 \| peak VRAM 0.91 GB |
| 2026-11-02 | `fusion_multistream_clip_frozen_s42` | `1c8a749` | kaggle | 0.3701 | 0.3757 | 12.7 | Phase 4 fusion_multistream_clip_frozen seed 42 \| peak VRAM 1.27 GB |
| 2026-11-02 | `fusion_multistream_ocr_frozen_s42` | `1c8a749` | kaggle | 0.3773 | 0.3760 | 4.7 | Phase 4 fusion_multistream_ocr_frozen seed 42 \| peak VRAM 0.90 GB |
| 2026-11-02 | `fusion_multistream_desc_frozen_s42` | `1c8a749` | kaggle | 0.3779 | 0.3827 | 5.7 | Phase 4 fusion_multistream_desc_frozen seed 42 \| peak VRAM 0.90 GB |
| 2026-11-02 | `text_only_frozen_s1` | `1c8a749` | kaggle | 0.3760 | 0.3650 | 2.6 | Phase 4 text_only_frozen seed 1 \| peak VRAM 0.89 GB |
| 2026-11-02 | `image_only_frozen_s1` | `1c8a749` | kaggle | 0.3713 | 0.3758 | 23.1 | Phase 4 image_only_frozen seed 1 \| peak VRAM 0.48 GB |
| 2026-11-02 | `fusion_concat_frozen_s1` | `1c8a749` | kaggle | 0.3918 | 0.4169 | 25.4 | Phase 4 fusion_concat_frozen seed 1 \| peak VRAM 1.26 GB |
| 2026-11-02 | `fusion_concat_ftlast_s1` | `1c8a749` | kaggle | 0.3874 | 0.3629 | 13.2 | Phase 4 fusion_concat_ftlast seed 1 \| peak VRAM 1.51 GB |
| 2026-11-02 | `text_only_frozen_cw_s1` | `1c8a749` | kaggle | 0.4219 | 0.3831 | 4.6 | Phase 4 text_only_frozen_cw seed 1 \| peak VRAM 0.89 GB |
| 2026-11-02 | `fusion_concat_frozen_reg_s1` | `1c8a749` | kaggle | 0.3721 | 0.3556 | 21.3 | Phase 4 fusion_concat_frozen_reg seed 1 \| peak VRAM 1.26 GB |
| 2026-11-02 | `fusion_concat_frozen_reg_cw_s1` | `1c8a749` | kaggle | 0.3941 | 0.3869 | 11.5 | Phase 4 fusion_concat_frozen_reg_cw seed 1 \| peak VRAM 1.26 GB |
| 2026-11-02 | `fusion_multistream_frozen_s1` | `1c8a749` | kaggle | 0.3712 | 0.3917 | 8.9 | Phase 4 fusion_multistream_frozen seed 1 \| peak VRAM 0.91 GB |
| 2026-11-02 | `fusion_multistream_clip_frozen_s1` | `1c8a749` | kaggle | 0.3694 | 0.3763 | 9.9 | Phase 4 fusion_multistream_clip_frozen seed 1 \| peak VRAM 1.27 GB |
| 2026-11-02 | `fusion_multistream_ocr_frozen_s1` | `1c8a749` | kaggle | 0.3746 | 0.3795 | 9.1 | Phase 4 fusion_multistream_ocr_frozen seed 1 \| peak VRAM 0.90 GB |
| 2026-11-02 | `fusion_multistream_desc_frozen_s1` | `1c8a749` | kaggle | 0.3879 | 0.3760 | 8.1 | Phase 4 fusion_multistream_desc_frozen seed 1 \| peak VRAM 0.90 GB |
| 2026-11-02 | `text_only_frozen_s2` | `1c8a749` | kaggle | 0.3659 | 0.3588 | 2.5 | Phase 4 text_only_frozen seed 2 \| peak VRAM 0.89 GB |
| 2026-11-02 | `image_only_frozen_s2` | `1c8a749` | kaggle | 0.3791 | 0.3774 | 21.6 | Phase 4 image_only_frozen seed 2 \| peak VRAM 0.48 GB |
| 2026-11-02 | `fusion_concat_frozen_s2` | `1c8a749` | kaggle | 0.3763 | 0.3689 | 23.9 | Phase 4 fusion_concat_frozen seed 2 \| peak VRAM 1.26 GB |
| 2026-11-02 | `fusion_concat_ftlast_s2` | `1c8a749` | kaggle | 0.3700 | 0.3746 | 12.1 | Phase 4 fusion_concat_ftlast seed 2 \| peak VRAM 1.50 GB |
| 2026-11-02 | `text_only_frozen_cw_s2` | `1c8a749` | kaggle | 0.3739 | 0.3702 | 2.7 | Phase 4 text_only_frozen_cw seed 2 \| peak VRAM 0.89 GB |
| 2026-11-02 | `fusion_concat_frozen_reg_s2` | `1c8a749` | kaggle | 0.3741 | 0.3700 | 12.6 | Phase 4 fusion_concat_frozen_reg seed 2 \| peak VRAM 1.26 GB |
| 2026-11-02 | `fusion_concat_frozen_reg_cw_s2` | `1c8a749` | kaggle | 0.3951 | 0.4013 | 12.2 | Phase 4 fusion_concat_frozen_reg_cw seed 2 \| peak VRAM 1.26 GB |
| 2026-11-02 | `fusion_multistream_frozen_s2` | `1c8a749` | kaggle | 0.3698 | 0.3677 | 5.2 | Phase 4 fusion_multistream_frozen seed 2 \| peak VRAM 0.91 GB |
| 2026-11-02 | `fusion_multistream_clip_frozen_s2` | `1c8a749` | kaggle | 0.4088 | 0.3770 | 27.7 | Phase 4 fusion_multistream_clip_frozen seed 2 \| peak VRAM 1.27 GB |
| 2026-11-02 | `fusion_multistream_ocr_frozen_s2` | `1c8a749` | kaggle | 0.4017 | 0.3883 | 12.3 | Phase 4 fusion_multistream_ocr_frozen seed 2 \| peak VRAM 0.90 GB |
| 2026-11-02 | `fusion_multistream_desc_frozen_s2` | `1c8a749` | kaggle | 0.3844 | 0.3868 | 8.5 | Phase 4 fusion_multistream_desc_frozen seed 2 \| peak VRAM 0.90 GB |

### Tóm tắt Phase 4 (mean ± std trên 3 seed, test = private test 1.504 mẫu)

| Config | Val macro-F1 | Test macro-F1 | Test macro-F1 3 lớp | Weighted-F1 | Accuracy | F1 not | F1 text | F1 image | F1 multi | Phút | VRAM (GB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `text_only_frozen` | 0.373 ± 0.006 | 0.365 ± 0.006 | 0.487 ± 0.008 | 0.736 ± 0.012 | 0.741 ± 0.013 | 0.651 | 0.000 | 0.000 | 0.809 | 3.00 | 0.89 |
| `image_only_frozen` | 0.366 ± 0.011 | 0.370 ± 0.011 | 0.494 ± 0.015 | 0.714 ± 0.013 | 0.715 ± 0.021 | 0.608 | 0.000 | 0.080 | 0.793 | 18.87 | 0.48 |
| `fusion_concat_frozen` | 0.378 ± 0.013 | 0.380 ± 0.033 | 0.506 ± 0.044 | 0.733 ± 0.019 | 0.736 ± 0.024 | 0.637 | 0.000 | 0.073 | 0.809 | 20.77 | 1.26 |
| `fusion_concat_ftlast` | 0.379 ± 0.009 | 0.370 ± 0.007 | 0.494 ± 0.009 | 0.732 ± 0.023 | 0.734 ± 0.030 | 0.646 | 0.000 | 0.032 | 0.804 | 19.90 | 1.51 |
| `text_only_frozen_cw` | 0.390 ± 0.028 | 0.381 ± 0.010 | 0.508 ± 0.013 | 0.746 ± 0.002 | 0.752 ± 0.004 | 0.650 | 0.000 | 0.050 | 0.824 | 3.30 | 0.89 |
| `fusion_concat_frozen_reg` | 0.381 ± 0.013 | 0.366 ± 0.009 | 0.488 ± 0.013 | 0.741 ± 0.024 | 0.750 ± 0.032 | 0.647 | 0.000 | 0.000 | 0.819 | 18.07 | 1.26 |
| `fusion_concat_frozen_reg_cw` | 0.399 ± 0.007 | **0.402 ± 0.015** | **0.535 ± 0.020** | 0.753 ± 0.008 | 0.755 ± 0.013 | 0.644 | 0.000 | 0.128 | 0.834 | 13.77 | 1.26 |
| `fusion_multistream_frozen` | 0.370 ± 0.001 | 0.377 ± 0.013 | 0.503 ± 0.017 | 0.743 ± 0.010 | 0.750 ± 0.014 | 0.650 | 0.000 | 0.039 | 0.820 | 8.03 | 0.91 |
| `fusion_multistream_clip_frozen` | 0.383 ± 0.023 | 0.376 ± 0.001 | 0.502 ± 0.001 | 0.755 ± 0.014 | 0.768 ± 0.020 | 0.653 | 0.000 | 0.016 | 0.836 | 16.77 | 1.27 |
| `fusion_multistream_ocr_frozen` | 0.385 ± 0.015 | 0.381 ± 0.006 | 0.508 ± 0.008 | 0.745 ± 0.012 | 0.748 ± 0.017 | 0.650 | 0.000 | 0.053 | 0.822 | 8.70 | 0.90 |
| `fusion_multistream_desc_frozen` | 0.383 ± 0.005 | 0.382 ± 0.005 | 0.509 ± 0.007 | 0.753 ± 0.017 | 0.761 ± 0.023 | 0.659 | 0.000 | 0.038 | 0.830 | 7.43 | 0.90 |

Ghi chú: macro-F1 3 lớp = macro-F1 tính trên `not-sarcasm` / `image-sarcasm` / `multi-sarcasm` (bỏ `text-sarcasm` vì test chỉ có 7 mẫu). Không config nào dự đoán được `text-sarcasm` (F1 = 0.000 ở cả 3 seed).
