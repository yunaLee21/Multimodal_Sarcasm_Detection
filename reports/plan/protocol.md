# Evaluation protocol (frozen 2026-11-23, pending team sign-off)

Applies to every result reported in the EACL 2027 SRW paper. Changes after sign-off are dated and listed at the bottom.

## Data roles

| Split | n | Role | History to disclose |
|---|---|---|---|
| Official train | 10,805 | Training; 3-fold stratified group CV for any new selection | — |
| Public test | — | Reported only | Used as the validation set for checkpoint selection in Phase 11 and the controlled runs (`data.split=official`: val = public test) |
| Private test | 1,504 (458 / 7 / 40 / 999) | Final configuration scored once; diagnostics with no selection | Viewed in Phases 10–11 and in the controlled comparison |

## Model selection (new runs only)

- 3-fold group CV on train: `data/processed/empirical/splits/folds_k3_s42.json`. Groups = shared caption or identical image.
- Selection metric: 4-class macro-F1 on the concatenated out-of-fold predictions. Report 3-class macro-F1 (without `text`) alongside it.
- No selection, tuning or threshold choice may use the public or private test.
- No threshold tuning for `text-sarcasm` (7 test samples). It is reported as a limitation.

## Statistics

- Per-seed results, mean ± SD over seeds 42 / 1 / 2.
- Paired bootstrap, B = 10,000. Every model and seed uses the same resampled ids. These intervals are conditional on the seeds.
- Per-class P / R / F1 and the confusion matrix for every headline model.
- Subsets with fewer than 30 samples (audit subsets): report counts and an exact test, not bootstrap CIs.

## Image-dependence diagnostics (WS-A)

- Inference only, on existing checkpoints. Results are never used to choose a model, τ or threshold.
- Neutral-image statistics (the mean image embedding) come from the train split only.
- Caption-only models are the control. Any prediction change under swap or mask invalidates the run.

## Human audit (WS-C)

- Raters are blind to benchmark labels, model predictions and each other. Passes: caption → image → pair.
- `image_contribution` is filled only when the rater judged the pair sarcastic; otherwise it is `NA`.
- AI raters are optional. Their results are reported separately and never mixed into human agreement.
- Audit subset definitions are fixed by rule (pair sarcastic, caption alone not sarcastic, `image_contribution = can`):
  *both humans*, *all four raters* and *either human*. Counts are reported as they come out.

## Changes after sign-off

(none)
