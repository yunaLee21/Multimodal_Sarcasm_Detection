# WS-A: image dependence of the Phase 11 checkpoints (private test)

Date: 2026-11-23 · Code: working tree on `b0a772c` + uncommitted `scripts/diagnose_image_dependence.py`,
`src/vimmsd/analysis/image_dependence.py` · Machine: local server, RTX 5070
Outputs: `data/server/outputs/p12_srw/image_dependence/` (`arm_condition.csv`, `per_run_condition.csv`,
`flip_matrix.csv`, `subsets.json`, `<run>/probs.npz`, `fig_flip_direction.{pdf,png}`, `run.log`)

Inference only. The private test has been viewed before (see `reports/plan/protocol.md`). Nothing here selects a model.

## Setup

- **Checkpoints:** 9 existing ones: cross-attn + CE, cross-attn + LA and caption-only + LA, each with seeds 42 / 1 / 2.
- **What changes:** the caption is fixed and only the image-encoder output (frozen CLIP ViT-B/32) is replaced.
- **Swaps** (5 swap seeds each, so 15 draws per arm):
  - random: a derangement of the 1,504 test images;
  - same-label: a derangement within each gold class;
  - different-label: a random image from another gold class.
- **Masks** (1 draw per seed):
  - mean: the train-mean CLIP output over 10,805 train images;
  - grey: a uniform grey image.
- **Validity checks:**
  - With the original images, every checkpoint reproduces its saved `test_predictions.csv`: 0 / 1,504 mismatches, max |Δp| = 3e-8.
  - Caption-only shows **0 flips under all 16 conditions**, as it must.
- **CIs:** the bootstrap resamples test ids (B = 10,000). Model seeds and swap draws are averaged per sample *before* resampling, so the CIs do **not** include seed variance. The seed spread is shown separately below.

## Full private test (n = 1,504)

Flip rate = share of predictions that change. Δp(gold) = mean change in the gold-class probability.

| Arm | Condition | Flip rate [95% CI] | Δp(gold) [95% CI] | Macro-F1 (4) orig → cond | Accuracy orig → cond | # predicted sarcastic, orig → cond |
|---|---|---|---|---|---|---|
| X-attn + LA | random swap | 0.159 [0.149, 0.169] | −0.033 [−0.038, −0.028] | 0.404 → 0.370 | 0.755 → 0.720 | 1098 → 1132 |
| | same-label swap | 0.143 [0.132, 0.153] | +0.001 [−0.004, +0.006] | 0.404 → 0.412 | 0.755 → 0.773 | 1098 → 1106 |
| | diff-label swap | 0.180 [0.169, 0.191] | −0.061 [−0.066, −0.056] | 0.404 → 0.344 | 0.755 → 0.677 | 1098 → 1093 |
| | mean mask | 0.355 [0.341, 0.369] | −0.071 [−0.080, −0.062] | 0.404 → 0.290 | 0.755 → 0.572 | 1098 → 632 |
| | grey mask | 0.319 [0.304, 0.335] | −0.073 [−0.082, −0.063] | 0.404 → 0.313 | 0.755 → 0.609 | 1098 → 651 |
| X-attn + CE | random swap | 0.168 [0.159, 0.176] | −0.029 [−0.035, −0.023] | 0.390 → 0.363 | 0.733 → 0.702 | 963 → 955 |
| | same-label swap | 0.160 [0.151, 0.169] | +0.003 [−0.003, +0.009] | 0.390 → 0.393 | 0.733 → 0.741 | 963 → 958 |
| | diff-label swap | 0.194 [0.185, 0.203] | −0.064 [−0.070, −0.057] | 0.390 → 0.342 | 0.733 → 0.657 | 963 → 900 |
| | mean mask | 0.177 [0.165, 0.189] | −0.038 [−0.046, −0.030] | 0.390 → 0.345 | 0.733 → 0.685 | 963 → 766 |
| | grey mask | 0.155 [0.144, 0.166] | −0.002 [−0.010, +0.006] | 0.390 → 0.348 | 0.733 → 0.705 | 963 → 871 |
| Caption-only + LA | all 16 | 0.000 | 0.000 | 0.384–0.396 (unchanged) | unchanged | unchanged |

**Seed spread is large for the masks.** Flip rate per model seed (42 / 1 / 2):

| Arm | Random swap | Mean mask |
|---|---|---|
| X-attn + LA | 0.177 / 0.159 / 0.140 | 0.367 / **0.625** / 0.073 |
| X-attn + CE | 0.205 / 0.070 / 0.228 | 0.199 / 0.078 / 0.253 |

**Reading:**
- The image matters to the cross-attention models. About one prediction in six changes when the image is swapped.
- A different-label image costs more than a random one (Δp(gold) −0.06 vs −0.03).
- A same-label image costs nothing, and even nudges macro-F1 up (0.404 → 0.412). The image carries label-correlated information that the model uses.
- Masking removes sarcasm predictions: LA goes from 1,098 to about 630 predicted sarcastic. Most mask flips go sarcastic → not (figure below).
- How strongly a model reacts to an *out-of-distribution* image (the mean or grey embedding) varies widely by seed (LA 0.07–0.63). Swap results are stable across seeds (LA 0.14–0.18).
- So the swap numbers are the reliable measure of image dependence. The mask numbers mostly measure how each seed extrapolates to an image unlike any it was trained on.

## Flip direction by gold class (Figure 1)

`fig_flip_direction.pdf`, numbers in `fig_flip_direction_table.csv`. `text` is omitted (n = 7).

| Arm | Gold | Random swap: sarc→not / not→sarc / between sarc | Mean mask: sarc→not / not→sarc / between sarc |
|---|---|---|---|
| X-attn + LA | not (458) | 0.053 / 0.151 / 0.013 | 0.205 / 0.023 / 0.009 |
| | image (40) | 0.160 / 0.078 / 0.025 | 0.383 / 0.008 / 0.017 |
| | multi (999) | 0.062 / 0.053 / 0.012 | 0.377 / 0.010 / 0.021 |
| X-attn + CE | not (458) | 0.055 / 0.099 / 0.008 | 0.120 / 0.011 / 0.003 |
| | image (40) | 0.137 / 0.050 / 0.035 | 0.267 / 0.008 / 0.017 |
| | multi (999) | 0.090 / 0.064 / 0.009 | 0.160 / 0.024 / 0.019 |

Under a random swap, gold-`not` samples move *toward* sarcasm (LA 15%), and gold-`image` samples move *away* from it (16%). A random image is evidence for or against sarcasm, depending on the sample. Flips between `image` and `multi` are rare (≤ 3.5%). The image shifts the sarcastic / not decision far more than it shifts *which kind* of sarcasm.

## Gate A: is the dependence concentrated where humans say the image is needed?

Audit subsets (the rule from the audit README; `subsets.json`): both humans 14, all four raters 11, either human 22,
all 40 pilot items. The source plan's "11 / 16" do not reproduce. **The main set is the 14 both-humans items**
(decided 2026-11-23); 11 and 22 are sensitivity checks with the same reading.

| Arm | Condition | Both humans (14) | Other 26 pilot items | Permutation p (14 vs 26) | Full test, gold `image` (40) |
|---|---|---|---|---|---|
| X-attn + LA | random swap | 0.338 | 0.287 | 0.46 | 0.263 |
| X-attn + LA | mean mask | 0.571 | 0.436 | 0.15 | 0.408 |
| X-attn + CE | random swap | 0.233 | 0.259 | 0.73 | 0.222 |
| X-attn + CE | mean mask | 0.357 | 0.128 | 0.014 | 0.292 |

(Per-sample flip rate averaged over seeds and draws. 10,000 label permutations within the 40 pilot items.
Four tests, uncorrected; only CE / mean mask has p < 0.05, and it would not survive Holm.)

**Reading:**
- The 14 image-needed items flip more than the full test (LA swap 0.34 vs 0.16).
- But the pilot was *sampled* for prediction instability (`image_flip`, `not_loss_all_seeds`, ... groups). Inside the pilot, the image-needed items flip about as often as the other 26 (0.34 vs 0.29, p = 0.46).
- The gold-`image` class on the full test flips at 0.26, close to the image-needed set.
- So the pilot gives **no evidence that the model's image dependence lines up with human "image needed" judgments**. The excess over the full test is explained by how the pilot was sampled and by class.
- With n = 14, the test can only detect large differences. A null here is weak evidence.

*Paper numbers:* `scripts/build_srw_tables.py` recomputes this table with the pre-registered round-2 test. That test
permutes within gold class and excludes the item with a blank pair rating. LA swap becomes 0.338 vs 0.293, p = .533; CE mask p = .379; Holm 1.0
for all four tests. The reading is unchanged.

**Gate A verdict (provisional, pending round-2 audit):**
- Claim 2 holds in its weak form: the image changes predictions measurably, swaps are stable across seeds, and the caption-only control is exactly null.
- The strong form, dependence concentrated on human image-needed items, is **not supported by the pilot**.
- The round-2 stratified random half (60 items) is the proper test, because it is not sampled for instability.
- Plan the paper as "the model uses the image, but not specifically where people say it should", unless round 2 shows otherwise.

## Out-of-fold train check (τ = 1.0 fold checkpoints, n = 10,805)

Models: `p12_srw/tau100_f{0,1,2}_s42`. Each is cross-attn + LA with τ = 1.0, the Phase 11 settings, trained on 2 of
3 group folds and evaluated on the held-out fold. Seed 42 only. The swaps are drawn within each held-out fold.
Outputs: `data/server/outputs/p12_srw/image_dependence_oof/`. The original condition reproduces every fold's
`val_predictions.csv` (0 mismatches).

The mask uses the same train-mean embedding as above (all 10,805 train images, including the held-out fold). Because it
is one constant vector, the leak is negligible, but it is not strictly fold-clean.

| Condition | Flip rate [95% CI] | Δp(gold) | Macro-F1 (4) orig → cond | Flip rate by gold: not / text / image / multi |
|---|---|---|---|---|
| random swap | 0.205 [0.199, 0.211] | −0.028 | 0.380 → 0.354 | 0.186 / 0.236 / 0.265 / 0.225 |
| same-label swap | 0.191 [0.185, 0.196] | −0.005 | 0.380 → 0.385 | 0.175 / 0.216 / 0.216 / 0.209 |
| diff-label swap | 0.217 [0.212, 0.223] | −0.051 | 0.380 → 0.334 | 0.200 / 0.210 / 0.256 / 0.238 |
| mean mask | 0.228 [0.220, 0.236] | +0.007 | 0.380 → 0.356 | 0.173 / 0.195 / 0.310 / 0.299 |
| grey mask | 0.209 [0.201, 0.217] | +0.019 | 0.380 → 0.348 | 0.161 / 0.208 / 0.303 / 0.266 |

**Reading:**
- The private-test swap pattern replicates on held-out train data with 442 `image` samples, instead of 40:
  - about one prediction in five changes under a random swap;
  - different-label > random > same-label, in both flip rate and Δp(gold);
  - a same-label image slightly helps macro-F1.
- Gold-`image` samples flip a bit more than gold-`not` (0.265 vs 0.186 under a random swap). The dependence is mildly larger for the class that is defined by the image, but far from exclusive to it.
- The masks are much milder here than on the private test for `ca_la` (0.23 vs 0.36). This confirms that mask effects are seed- and model-specific, and that the swaps are the measure to report.

## Not done yet

- Round-2 audit items (after WS-C ratings).
- Same-source / different-source swap: dropped (only 229 / 1,504 captions carry a source tag).
