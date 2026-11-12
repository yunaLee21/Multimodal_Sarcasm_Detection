# Empirical study — Phase 1: infrastructure

Phase 1 of `docs/empirical_analysis_plan.md` (section 13), 12 November 2026. Covers implementation items 1–6 of plan section 12; item 7 (the final empirical report builder) is left for Phase 4. 244 tests pass. No Experiment A–E training has run yet.

| plan item | what exists | how to use |
|---|---|---|
| 1. Split manifest + validator | `scripts/build_empirical_splits.py`, `src/vimmsd/data/splits.py` (Phase 0); the loader now reads the manifest | `data.fold.manifest=<json> data.fold.index=<i> train.eval_test=false`: train = other folds, val = held-out fold. Refuses a manifest built from a different train file, a missing id, or private-test scoring. |
| 2. Augmentation | `scripts/augment_minority.py` (generate / sheet / import), `src/vimmsd/data/augment_review.py`, `docs/augmentation_review_guide.md` | `data.augmentation.manifest=<jsonl>` adds accepted variants to the training portion only; each variant keeps its parent's image, OCR, description, label, group and fold, with `parent_id`. |
| 3. Sampling exponent + step budget | `data.sampler=power`, `data.sampler_power`, `train.steps_per_epoch`; `exposure.json` in every run | B2 = `power` with 0.5. `steps_per_epoch` fixes optimizer updates per epoch for every treatment (C screening: 451). Each run logs draws, updates, unique ids and unique parents. |
| 4. Input diagnostics | `scripts/diagnose_inputs.py`, `src/vimmsd/analysis/interventions.py` | Separate pixel / OCR / description / joint replacement and OCR / description removal; 5 fixed donor permutations shared by all models and seeds. |
| 5. Statistics | `scripts/empirical_stats.py`, `src/vimmsd/analysis/stats.py` | `contrasts` (paired bootstrap, 2,000 replicates, optional group resampling, Holm), `pooled` (out-of-fold over folds), `subsets` (fixed val subsets with support). |
| 6. Plans | `configs/experiments/empirical_A.yaml` (21 runs), `empirical_C_screen.yaml` (B0–B3 × 3 folds) | `refresh_report: false` keeps the CoT experiment report untouched. |

## Checks run on real data

- **Fold manifest**: folds of 7,203–7,204 train / 3,601–3,602 val, no id overlap; the guard rejects `train.eval_test=true`.
- **Sampler and budget** (1-epoch smoke run, fold 0, B2 settings): exactly 451 updates and 7,216 draws; 4,302 unique samples, because square-root sampling repeats minority samples.
- **Augmentation import** (fake reviews in a throwaway folder): disagreements rejected, at most one variant per parent, about two-thirds of variants enter each fold's training portion, and no variant or variant parent appears in that fold's val.
- **Statistics**: on the earlier original-vs-clean runs, `contrasts` reproduces the known means (0.3932 vs 0.3922) and gives 95% intervals that include zero.
- **Diagnostics** on `distill_flat_a_s42` (= M, seed 42, 1 permutation): the unmodified model reproduces its recorded val macro-F1 (0.3802); prediction flip rates are 7.3% for pixels only, 4.1% OCR only, 3.3% description only and 14.4% joint. These numbers are a smoke check, not Experiment B results.

## Augmentation status (plan gate: "reject invalid augmentation before training")

Generated with `vinai/vinai-translate-vi2en-v2` → `vinai/vinai-translate-en2vi-v2` (beam 5 to English, top-p 0.9 sampling back, seed 42), 2 candidates for each of the 519 minority parents (77 `text-sarcasm`, 442 `image-sarcasm`): 1,038 candidates, 72 rejected automatically (49 unchanged, 20 duplicate candidates, 2 duplicates of train captions, 1 empty), **966 to review**.

A first look shows the risk the plan anticipated: literal back-translation often removes the joke (for example, the pun in "cơm nắm đấm" becomes plain "cơm"). The review sheets are `data/processed/empirical/augmentation/review_reviewer1.csv` and `review_reviewer2.csv`; instructions are in `docs/augmentation_review_guide.md`. **B4/B5 stay out of the screening plan until the review is imported.** If too few candidates keep their label, B4/B5 are dropped and the audit is reported (plan section 8.2).

## Ready for Phase 2

Experiment A (`empirical_A.yaml`) and the B0–B3 screening (`empirical_C_screen.yaml`) can run now; neither depends on the augmentation review. Estimated GPU time is about 2 h and 1.5 h (Phase 0, section 5).
