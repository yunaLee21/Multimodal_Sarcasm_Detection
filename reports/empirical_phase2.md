# Empirical study — Phase 2: modality matrix, input diagnostics, imbalance screening

Phase 2 of `docs/empirical_analysis_plan.md`, run 12 November 2026 on the GPU server (RTX 5070) at commit `ef3dcfc`, protocol as frozen in `reports/empirical_phase0.md` §3. 33 training runs, all finished without errors (15:01–18:41 UTC). Private test was not scored. Raw outputs: `data/server/outputs/empA_*`, `empC_*`, and `data/server/outputs/_empirical/{statistics,diagnostics}/`.

**Uncertainty.** Intervals are paired bootstraps (2,000 replicates) over evaluation examples with the trained models fixed; they do not include training randomness. "Group" intervals resample caption/image groups instead of examples. Three seeds describe variability but are weak evidence for small differences (plan §11).

## 1. Experiment A: modality contributions (val = public test, 1,413 samples)

Mean ± SD over seeds 1, 2, 42. Checkpoints selected on macro-F1 without `text-sarcasm`. No model ever predicted the single val `text-sarcasm` sample, so four-class macro-F1 is exactly ¾ of the three-class score here.

| model | inputs | macro-F1 (4 classes) | macro-F1 (3 classes) | weighted-F1 | F1 `image-sarcasm` (28 samples) |
|---|---|---|---|---|---|
| T | caption | 0.381 ± 0.007 | 0.507 ± 0.009 | 0.738 | 0.070 |
| I | image pixels | 0.346 ± 0.008 | 0.461 ± 0.010 | 0.711 | 0.000 |
| C | caption + CLIP (regularized concat) | **0.399 ± 0.008** | **0.532 ± 0.010** | 0.746 | **0.146** |
| O | caption + OCR | 0.383 ± 0.014 | 0.511 ± 0.018 | 0.768 | 0.035 |
| D | caption + description | 0.373 ± 0.002 | 0.497 ± 0.002 | 0.760 | 0.000 |
| OD | caption + OCR + description | 0.387 ± 0.013 | 0.515 ± 0.017 | 0.737 | 0.113 |
| M | caption + OCR + description + CLIP | 0.391 ± 0.011 | 0.521 ± 0.015 | 0.760 | 0.077 |

**Predeclared contrasts** (four-class macro-F1; Holm within the family of six):

| contrast | Δ (mean of seeds) | per seed (1, 2, 42) | 95% CI | p | p Holm | group 95% CI |
|---|---|---|---|---|---|---|
| C − T | **+0.019** | +0.013 +0.030 +0.013 | [−0.006, +0.042] | 0.13 | 0.79 | [−0.000, +0.043] |
| O − T | +0.003 | +0.011 −0.002 −0.001 | [−0.012, +0.015] | 0.63 | 1.00 | [−0.009, +0.018] |
| D − T | −0.008 | −0.018 −0.001 −0.005 | [−0.023, +0.007] | 0.36 | 1.00 | [−0.030, +0.014] |
| OD − O | +0.003 | +0.002 +0.003 +0.005 | [−0.011, +0.019] | 0.70 | 1.00 | [−0.016, +0.024] |
| OD − D | +0.014 | +0.030 +0.003 +0.009 | [−0.008, +0.039] | 0.25 | 1.00 | [−0.015, +0.048] |
| M − OD | +0.005 | +0.002 +0.013 −0.001 | [−0.013, +0.023] | 0.56 | 1.00 | [−0.019, +0.027] |

Reading: adding CLIP image features to the caption (C − T) is the only contrast that is positive for every seed and exceeds the predeclared practical target of +1 point; its intervals still include zero, so it is suggestive, not established. OCR and descriptions as separate PhoBERT streams add nothing detectable over the caption (O − T, D − T near zero), and adding CLIP on top of them (M − OD) adds little. Most of C's gain shows up in `image-sarcasm` F1 (0.07 → 0.15, on 28 samples).

**Fixed val subsets** (descriptive; macro-F1 over the classes present; full table in `_empirical/statistics/subsets_*.md`): on the 56 val samples whose caption occurs elsewhere in val with a different label, C (0.588) and OD (0.623) are well above T (0.490), consistent with image information mattering most when the caption alone cannot decide; the subset is small. Samples without OCR text (214, mostly `not-sarcasm`) are easy for every model (0.66–0.76).

## 2. Experiment B: input dependence of C and M (no retraining)

Same val samples, caption and ids fixed, 5 shared random donor permutations without self-donors; mean over the 3 seeds (range across seeds in brackets). Flip rate = share of predictions that change; TV = mean total-variation shift of the probability vector. Per-class rates for `text-sarcasm` rest on one val sample and are omitted.

| model | intervention | flip rate | TV shift | flip rate by original class: not / image / multi |
|---|---|---|---|---|
| C | replace pixels | 0.199 [0.173–0.229] | 0.191 | 0.229 / 0.362 / 0.180 |
| M | replace pixels only | 0.090 [0.073–0.121] | 0.085 | 0.115 / 0.148 / 0.077 |
| M | replace OCR only | 0.096 [0.042–0.172] | 0.097 | 0.115 / 0.160 / 0.085 |
| M | replace description only | 0.068 [0.031–0.130] | 0.064 | 0.083 / 0.079 / 0.060 |
| M | replace pixels + OCR + description (one donor) | 0.183 [0.140–0.269] | 0.172 | 0.230 / 0.276 / 0.158 |
| M | remove OCR | 0.086 [0.064–0.126] | 0.096 | 0.099 / 0.143 / 0.079 |
| M | remove description | 0.067 [0.038–0.115] | 0.070 | 0.079 / 0.083 / 0.062 |

Reading: C's predictions depend on the image substantially (one in five flips; macro-F1 against the original labels falls from 0.399 to 0.342 on average), and most for `image-sarcasm` samples. M spreads its reliance over three image-derived streams: each alone moves 7–10% of predictions and all three together about as much as pixels move C. M's sensitivity varies a lot by seed (seed 2 is two to four times as sensitive as seeds 1 and 42), so single-seed diagnostics would mislead. These are disruption measures under arbitrary, partly implausible substitutions, not accuracy on relabeled data (plan §7). The M seed-42 run reproduces the earlier `distill_flat_a_s42` exactly (val macro-F1 0.3802 and identical flip rates).

## 3. Experiment C screening: imbalance treatments on C (internal grouped folds)

C × 3 grouped folds × seed 42, 451 optimizer steps per epoch for every treatment, checkpoints selected on each fold's four-class macro-F1. Pooled out-of-fold over all 10,805 train samples:

| treatment | pooled macro-F1 | per fold | F1 not | F1 text (77) | F1 image (442) | P / R image | F1 multi |
|---|---|---|---|---|---|---|---|
| B0 CE | 0.381 | 0.346 0.387 0.400 | 0.722 | 0.000 | 0.171 | 0.30 / 0.12 | 0.633 |
| B1 CE + √ inverse class weights | **0.389** | 0.375 0.390 0.401 | 0.713 | 0.000 | 0.199 | 0.22 / 0.18 | 0.646 |
| B2 √ inverse-frequency sampling | 0.385 | 0.386 0.380 0.392 | 0.707 | 0.000 | 0.204 | 0.17 / 0.25 | 0.629 |
| B3 focal, γ = 2 | 0.371 | 0.376 0.365 0.360 | 0.719 | 0.000 | 0.145 | 0.23 / 0.11 | 0.619 |

Against B0 (paired bootstrap over out-of-fold samples, one training seed):

| contrast | Δ macro-F1 | 95% CI | group 95% CI |
|---|---|---|---|
| B1 − B0 | +0.008 | [−0.004, +0.019] | [−0.007, +0.024] |
| B2 − B0 | +0.004 | [−0.008, +0.016] | [−0.014, +0.023] |
| B3 − B0 | −0.011 | [−0.023, +0.001] | [−0.032, +0.015] |

Reading: no treatment recovers `text-sarcasm` at all (F1 0 for all 77 samples in every treatment). B1 and B2 raise `image-sarcasm` recall (0.12 → 0.18 / 0.25) at the cost of precision and `not-sarcasm` recall, a minority-focused tradeoff rather than a clear overall gain; B1's +0.008 is below the +1-point practical target and its interval includes zero. Focal loss is worst.

**Selection (plan §8.4).** Among the new treatments available so far (B2, B3), none beats B1, so the rule retains **B1** as the candidate for Phase 3 confirmation. This is provisional: B4/B5 (reviewed augmentation) are part of the screen but wait on the two-reviewer review of `data/processed/empirical/augmentation/review_reviewer{1,2}.csv`. If the review is completed, B4/B5 are screened on the same folds and budget before the choice is final.

## 4. Notes and deviations

- Run times: 1–19 min per run (image models 4–19 min, caption-only streams 1–5 min); total 3 h 25 min training plus 15 min diagnostics.
- The B0 fold-0 score (0.346) is notably lower than folds 1–2; fold-to-fold variation is of the same order as the treatment differences, which is why the pooled out-of-fold score and the group bootstrap are used for selection.
- No protocol amendments were made. `reports/experiment_report.md` (CoT) was not touched; the 33 runs were appended to `reports/results.md`.

## 5. Next (Phase 3)

Following plan §8.4, §9 and §10: refit C with B1 on the full official train (3 runs; C-CE already exists from Experiment A), retrain T and C with source tags removed (6 runs), and transfer B1 to T and M (6 runs), then repeat the diagnostics on the selected model. B4/B5 screening (6 runs) is added first if the augmentation review is done.
