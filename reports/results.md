# Kết quả thí nghiệm

Mỗi lần train xong, `scripts/train.py` in ra 1 dòng kết quả (và ghi vào `<output_dir>/<tên config>/results_row.md`). Chạy local thì dòng được tự thêm vào đây; chạy Kaggle/Colab thì copy dòng đó vào đây rồi commit.

| Ngày | Config | Commit | Môi trường | Val macro-F1 | Test macro-F1 | Thời gian train (phút) | Ghi chú |
|---|---|---|---|---|---|---|---|
| 2026-10-30 | `distill_flat_a_s42` | `e082390` | server | 0.3802 | nan | 5.3 | plan cot_distill_stage0 | peak VRAM 1.24 GB |
| 2026-10-30 | `distill_flat_b_s42` | `e082390` | server | 0.4041 | nan | 8.1 | plan cot_distill_stage0 | peak VRAM 1.25 GB |
| 2026-10-30 | `distill_hier_a_s42` | `e082390` | server | 0.3818 | nan | 9.3 | plan cot_distill_stage0 | peak VRAM 1.25 GB |
| 2026-10-30 | `distill_hier_b_s42` | `e082390` | server | 0.3778 | nan | 5.4 | plan cot_distill_stage0 | peak VRAM 1.27 GB |
| 2026-10-30 | `hier_cascade_s42` | `e082390` | server | 0.3800 | nan | 5.4 | plan cot_distill_stage0 | peak VRAM 1.25 GB |
| 2026-10-30 | `distill_flat_a_s1` | `e082390` | server | 0.4030 | nan | 8.0 | plan cot_distill_stage0 | peak VRAM 1.24 GB |
| 2026-10-30 | `distill_flat_b_s1` | `e082390` | server | 0.3764 | nan | 6.7 | plan cot_distill_stage0 | peak VRAM 1.25 GB |
| 2026-10-30 | `distill_hier_a_s1` | `e082390` | server | 0.3759 | nan | 5.4 | plan cot_distill_stage0 | peak VRAM 1.25 GB |
| 2026-10-30 | `distill_hier_b_s1` | `e082390` | server | 0.3836 | nan | 10.8 | plan cot_distill_stage0 | peak VRAM 1.27 GB |
| 2026-10-30 | `hier_cascade_s1` | `e082390` | server | 0.3741 | nan | 5.4 | plan cot_distill_stage0 | peak VRAM 1.25 GB |
| 2026-10-30 | `distill_flat_a_s2` | `e082390` | server | 0.3900 | nan | 16.0 | plan cot_distill_stage0 | peak VRAM 1.24 GB |
| 2026-10-30 | `distill_flat_b_s2` | `e082390` | server | 0.3855 | nan | 5.4 | plan cot_distill_stage0 | peak VRAM 1.25 GB |
| 2026-10-30 | `distill_hier_a_s2` | `e082390` | server | 0.3804 | nan | 10.7 | plan cot_distill_stage0 | peak VRAM 1.25 GB |
| 2026-10-30 | `distill_hier_b_s2` | `e082390` | server | 0.3896 | nan | 10.8 | plan cot_distill_stage0 | peak VRAM 1.27 GB |
| 2026-10-30 | `hier_cascade_s2` | `e082390` | server | 0.3857 | nan | 8.0 | plan cot_distill_stage0 | peak VRAM 1.25 GB |
| 2026-11-02 | `distill_flat_a_notags_s42` | `423c79f` | server | 0.3700 | nan | 5.4 | plan no_source_tags | peak VRAM 1.24 GB |
| 2026-11-02 | `distill_flat_a_notags_s1` | `423c79f` | server | 0.3919 | nan | 7.9 | plan no_source_tags | peak VRAM 1.24 GB |
| 2026-11-02 | `distill_flat_a_notags_s2` | `423c79f` | server | 0.3649 | nan | 5.4 | plan no_source_tags | peak VRAM 1.24 GB |
| 2026-11-03 | `b2_text_f0_s42` | `2e34aa4` | server | 0.3813 | nan | 1.5 | plan b2_stacking | peak VRAM 0.87 GB |
| 2026-11-03 | `b2_text_f1_s42` | `2e34aa4` | server | 0.3744 | nan | 0.9 | plan b2_stacking | peak VRAM 0.87 GB |
| 2026-11-03 | `b2_text_f2_s42` | `2e34aa4` | server | 0.3739 | nan | 0.9 | plan b2_stacking | peak VRAM 0.87 GB |
| 2026-11-03 | `b2_text_f3_s42` | `2e34aa4` | server | 0.3805 | nan | 0.7 | plan b2_stacking | peak VRAM 0.87 GB |
| 2026-11-03 | `b2_text_f4_s42` | `2e34aa4` | server | 0.3755 | nan | 1.1 | plan b2_stacking | peak VRAM 0.87 GB |
| 2026-11-03 | `b2_text_full_s42` | `2e34aa4` | server | 0.3778 | nan | 1.6 | plan b2_stacking | peak VRAM 0.87 GB |
| 2026-11-03 | `b2_image_f0_s42` | `2e34aa4` | server | 0.3586 | nan | 5.8 | plan b2_stacking | peak VRAM 1.06 GB |
| 2026-11-03 | `b2_image_f1_s42` | `2e34aa4` | server | 0.3510 | nan | 3.9 | plan b2_stacking | peak VRAM 1.06 GB |
| 2026-11-03 | `b2_image_f2_s42` | `2e34aa4` | server | 0.3555 | nan | 4.8 | plan b2_stacking | peak VRAM 1.06 GB |
| 2026-11-03 | `b2_image_f3_s42` | `2e34aa4` | server | 0.3572 | nan | 6.7 | plan b2_stacking | peak VRAM 1.06 GB |
| 2026-11-03 | `b2_image_f4_s42` | `2e34aa4` | server | 0.3590 | nan | 3.9 | plan b2_stacking | peak VRAM 1.06 GB |
| 2026-11-03 | `b2_image_full_s42` | `2e34aa4` | server | 0.3500 | nan | 4.7 | plan b2_stacking | peak VRAM 1.06 GB |
| 2026-11-03 | `b2_multi_f0_s42` | `2e34aa4` | server | 0.3689 | nan | 9.9 | plan b2_stacking | peak VRAM 1.24 GB |
| 2026-11-03 | `b2_multi_f1_s42` | `2e34aa4` | server | 0.3866 | nan | 11.1 | plan b2_stacking | peak VRAM 1.24 GB |
| 2026-11-03 | `b2_multi_f2_s42` | `2e34aa4` | server | 0.3717 | nan | 5.5 | plan b2_stacking | peak VRAM 1.24 GB |
| 2026-11-03 | `b2_multi_f3_s42` | `2e34aa4` | server | 0.3928 | nan | 8.8 | plan b2_stacking | peak VRAM 1.24 GB |
| 2026-11-03 | `b2_multi_f4_s42` | `2e34aa4` | server | 0.3912 | nan | 9.9 | plan b2_stacking | peak VRAM 1.24 GB |
| 2026-11-03 | `b2_multi_full_s42` | `2e34aa4` | server | 0.3802 | nan | 5.3 | plan b2_stacking | peak VRAM 1.24 GB |
| 2026-11-10 | `text_only_orig_s42` | `b25b4e1` | server | 0.3894 | 0.3893 | 4.5 | plan denoise_compare | peak VRAM 3.31 GB |
| 2026-11-10 | `text_only_clean_s42` | `b25b4e1` | server | 0.4075 | 0.3917 | 8.0 | plan denoise_compare | peak VRAM 3.31 GB |
| 2026-11-10 | `image_only_orig_s42` | `b25b4e1` | server | 0.3490 | 0.3544 | 6.7 | plan denoise_compare | peak VRAM 1.65 GB |
| 2026-11-10 | `image_only_clean_s42` | `b25b4e1` | server | 0.3450 | 0.3194 | 13.4 | plan denoise_compare | peak VRAM 1.65 GB |
| 2026-11-10 | `text_only_orig_s1` | `b25b4e1` | server | 0.4066 | 0.3926 | 8.0 | plan denoise_compare | peak VRAM 3.34 GB |
| 2026-11-10 | `text_only_clean_s1` | `b25b4e1` | server | 0.3913 | 0.3616 | 8.0 | plan denoise_compare | peak VRAM 3.31 GB |
| 2026-11-10 | `image_only_orig_s1` | `b25b4e1` | server | 0.3366 | 0.3375 | 6.7 | plan denoise_compare | peak VRAM 1.65 GB |
| 2026-11-10 | `image_only_clean_s1` | `b25b4e1` | server | 0.3471 | 0.3515 | 6.7 | plan denoise_compare | peak VRAM 1.65 GB |
| 2026-11-10 | `text_only_orig_s2` | `b25b4e1` | server | 0.3806 | 0.3759 | 4.5 | plan denoise_compare | peak VRAM 3.31 GB |
| 2026-11-10 | `text_only_clean_s2` | `b25b4e1` | server | 0.3808 | 0.3754 | 4.5 | plan denoise_compare | peak VRAM 3.31 GB |
| 2026-11-10 | `image_only_orig_s2` | `b25b4e1` | server | 0.3443 | 0.3469 | 5.3 | plan denoise_compare | peak VRAM 1.65 GB |
| 2026-11-10 | `image_only_clean_s2` | `b25b4e1` | server | 0.3549 | 0.3546 | 5.3 | plan denoise_compare | peak VRAM 1.65 GB |
