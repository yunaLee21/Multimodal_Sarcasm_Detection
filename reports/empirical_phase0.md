# Empirical study — Phase 0: audit, frozen protocol, splits, runtime

Phase 0 of `docs/empirical_analysis_plan.md` (section 13). Written 12 November 2026 on the GPU server (RTX 5070 12 GB, conda `tlenv-sarcasm`). No Experiment A–E training has run yet.

## 1. Data used

The main study uses the **original official data** (plan section 4.1). The `data/server/*` symlinks were pointed back to it on 12 November; the denoised copy at `/data/Phat/vimmsd-clear` is kept only as a separate, named intervention (its comparison is reported as mixed in the plan, section 3).

| split | samples | `json` sha256 (first 12) | labels |
|---|---|---|---|
| train | 10,805 | `8458f993ea06` | not 6,062 · text 77 · image 442 · multi 4,224 |
| public test (val) | 1,413 | `8905dc2e86af` | not 442 · text 1 · image 28 · multi 942 |
| private test | 1,504 | `b70bc3979b8c` | not 458 · text 7 · image 40 · multi 999 |

Full checksums (json, label file, digest of every image md5) are in `data/processed/empirical/splits/folds_k3_s42.json` → `meta.checksums`.

**Overlap across official splits** (normalized caption or image md5; `overlap.json`). No sample in a test split duplicates a train sample on both caption and image.

| pair | same caption | same image | either | both |
|---|---|---|---|---|
| train ↔ public test (1,413) | 54 | 5 | 59 (4.2%) | 0 |
| train ↔ private test (1,504) | 71 | 12 | 83 (5.5%) | 0 |
| public ↔ private test (1,504) | 42 | 3 | 45 (3.0%) | 0 |

The internal grouping below prevents leakage inside development folds; it does not change this overlap in the official benchmark.

## 2. Internal 3-fold split (plan section 4.2)

- **Groups**: union-find over train samples linked by the same normalized caption (html.unescape, NFC, whitespace collapse, strip, lowercase) **or** the same image md5, transitively. 8,621 groups; 705 have ≥ 2 samples; the largest has 34 (the `:))` caption, labels multi 12 / not 11 / image 11).
- **Splitter**: `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)`.
- **Validation** (passed): every id in exactly one fold, no group spans two folds, minimum support `text-sarcasm` ≥ 15 and `image-sarcasm` ≥ 100 per fold, every other class ≥ 1.

| fold | not | text | image | multi | total |
|---|---|---|---|---|---|
| 0 | 2,021 | 26 | 147 | 1,408 | 3,602 |
| 1 | 2,020 | 25 | 148 | 1,408 | 3,601 |
| 2 | 2,021 | 26 | 147 | 1,408 | 3,602 |

Manifest: `data/processed/empirical/splits/folds_k3_s42.json` (`folds`, `groups`, `meta`). Built by `scripts/build_empirical_splits.py`; the builder exits with an error instead of falling back to an ungrouped split. The loader cannot read this manifest yet: the existing `data.fold` uses the ungrouped `fold_of` (B2 stacking). Reading the manifest is Phase 1 work.

## 3. Frozen protocol

| item | setting |
|---|---|
| encoders | frozen PhoBERT (`vinai/phobert-base`) and CLIP (`openai/clip-vit-base-patch32`), flat MLP head, `image_resize: pad` |
| loss (Experiment A) | plain CE, no class weights, `sampler: random` |
| seeds | 1, 2, 42 for every paired comparison |
| training budget | `lr_head 1e-3`, max 20 epochs, early stopping patience 3, batch 16, warmup 0.1 (inherited by all seven configs) |
| checkpoint selection (official val) | `train.select_metric=macro_f1`, `train.select_exclude_labels=[text-sarcasm]` |
| private test | `train.eval_test=false` during development; one fixed final benchmark in Phase 4 |
| text cache | `text_preprocessed.json`, `pipeline_key 2c3852f6` (lowercase off, teencode on, emoji demojize, VnCoreNLP) |
| image text | `image_preprocessed.json` in `/data/Phat/vimmsd_cache` (OCR + Vintern descriptions), unchanged since 30 October |
| this machine | `train.num_workers=0` (`/dev/shm` is 64 MB) |
| prior correction | not part of the core matrix |

**Model ↔ config map** (all existing configs, no new architecture):

| plan ID | config | inputs |
|---|---|---|
| T | `text_only_frozen` | caption → PhoBERT |
| I | `image_only_frozen` | pixels → CLIP |
| C | `fusion_concat_frozen_reg` | caption + CLIP, regularized concat (proj 256, image feature dropout 0.2, image drop 0.3) |
| O | `fusion_multistream_ocr_frozen` | caption + OCR, separate PhoBERT streams |
| D | `fusion_multistream_desc_frozen` | caption + description |
| OD | `fusion_multistream_frozen` | caption + OCR + description |
| M | `fusion_multistream_clip_frozen` | caption + OCR + description + CLIP |

Note: C is the regularized concat with **plain CE**. The historical best (0.402 private macro-F1) is `fusion_concat_frozen_reg_cw`, i.e. C with square-root class weights, which corresponds to treatment B1 in Experiment C, not to C itself.

## 4. Audit of historical runs

Resolved `config.json` of every local run (56 in `/data/Phat/vimmsd_outputs` and the denoise outputs) was compared field by field with the planned configs plus the protocol overrides above, ignoring paths, run name, seed, worker count and evaluation-only settings.

| source | runs | verdict |
|---|---|---|
| Phase 4 on Kaggle (commit `1c8a749`, 2 November): all seven configs × seeds 42/1/2 | 33 | **Not reusable.** Checkpoints were selected on four-class macro-F1 (no `text-sarcasm` exclusion), private test was scored every run, and checkpoints / per-example probabilities are not on this machine. Keep as historical numbers. |
| `distill_flat_a_s{42,1,2}` (commit `e082390`) | 3 | **Compatible with M** (identical config, selection excludes `text-sarcasm`, `eval_test=false`, `best.pt` + `val_predictions.csv` present). |
| `b2_multi_full_s42` (commit `2e34aa4`) | 1 | Same config and seed as `distill_flat_a_s42`, and reproduces it exactly (val macro-F1 0.3802 both), which supports determinism across these commits. |
| `b2_text_full_s42` (commit `2e34aa4`) | 1 | **Compatible with T** for seed 42 only. |
| CoT `distill_*` (b/c/d), `hier_*`, `b2_*` folds, denoise comparison | rest | Not part of the plan's matrix (different heads, losses, folds or training data). |

Code changes since 30 October touch only cross-attention (not in the core matrix), tracking, and new options that default to off (`data.fold`, `data.caption_filter`, source-tag helpers); the text cache keeps the same `pipeline_key`. So the compatible runs above remain valid under the current code.

**Recommendation:** rerun all 21 Experiment A runs anyway. They are cheap on this GPU (section 5), and a single commit and output format for every cell is simpler to defend than mixing three commits. The four compatible runs then serve as a reproducibility check, not as substitutes.

### Issues found

1. **Phase 4 results were dropped from `reports/results.md`.** The 37 Kaggle rows and their summary table (added in `3f8783d`) were lost when GitHub merge `38374bb` (10 November) resolved a conflict in favour of the branch version; `origin/main` has the same loss. They are recoverable from `git show 3f8783d:reports/results.md`.
2. `configs/base.yaml` has `train.eval_test: true`, so every run scores private test unless overridden; the empirical plans must set `train.eval_test=false`.
3. The existing `sampler: weighted` is full inverse-frequency balancing (weight `1/count`), not the square-root rule required for treatment B2. B2 needs a new sampling exponent (Phase 1). `loss.class_weight: sqrt_inverse` already implements B1.
4. Running `run_experiments.py` regenerates `reports/experiment_report.md` (the CoT report) from whatever `data/server/outputs` points to. Empirical plans should not overwrite it (plan section 12, item 7).

## 5. Runtime and compute estimate

Measured run: C, seed 42, all protocol settings above (`empirical_p0_timing_C_s42`, commit `b6d294e`). 676 steps per epoch, about **65 s per epoch**, early stop after 6 epochs: **6.5 min** training, peak VRAM 1.23 GB, val macro-F1 0.391 (four-class; selected on macro-F1 without `text-sarcasm`). A run that used all 20 epochs would take about 22 min.

Estimate for the remaining work, scaling the other configs by their Phase 4 run times relative to C. Caption-only streams (T, O, D, OD) skip image decoding, so they are about 2–3× faster. These are estimates from one measured run, not medians; Phase 1 should replace them with the median of the first Experiment A runs (plan section 13).

| block | runs | typical | if every run hits 20 epochs |
|---|---|---|---|
| Experiment A (T, I, C, O, D, OD, M × 3 seeds) | 21 | ~2 h | ~5 h |
| Experiment C screening (C, 6 treatments × 3 folds, 2/3 of train) | 18 | ~1.5 h | ~4.5 h |
| C confirmation, D tag retraining, E transfer | up to 18 | ~2 h | ~5 h |
| **core total** | **up to 57** | **~5–6 h** | **~15 h** |

GPU time is not the bottleneck. Augmentation review time (B4/B5) is the open item for scheduling.

## 6. Gate

Phase 0 gate (plan section 13): "Stop and repair any leakage or invalid split." The grouped split is valid, and no test sample duplicates a train sample on both caption and image. The cross-split overlaps in section 1 are properties of the official benchmark and are reported, not repaired. **Gate passed**, subject to the issues in section 4 being handled in Phase 1.
