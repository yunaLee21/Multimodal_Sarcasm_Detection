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
