# Empirical study — Phase 3: B1 confirmation, source cues, transfer

Phase 3 of `docs/empirical_analysis_plan.md` (§8.4, §9, §10), run 13 November 2026 at commit `829018e`, same protocol as Phase 2 (`reports/empirical_phase0.md` §3). 15 training runs (03:43–05:21 UTC), all finished; then diagnostics and statistics. Private test was not scored. Raw outputs: `data/server/outputs/emp{B1,D,E}_*` and `_empirical/{statistics,diagnostics}/`.

Treatment selected in Phase 2: **B1**, cross-entropy with square-root inverse-frequency class weights (provisional; B4/B5 still wait on the augmentation review). All contrasts are paired by seed (1, 2, 42) on val (public test, 1,413 samples); intervals are paired bootstraps over evaluation examples (2,000 replicates), group intervals resample caption/image groups; neither includes training randomness.

## 1. Results on val

| model | macro-F1 (4 cl.) | weighted-F1 | accuracy | F1 not | F1 image (28) | recall image | F1 multi |
|---|---|---|---|---|---|---|---|
| T (caption, CE) | 0.381 ± 0.007 | 0.738 | 0.741 | 0.654 | 0.070 | 0.05 | 0.799 |
| **T + B1** | **0.412 ± 0.013** | 0.752 | 0.753 | 0.649 | 0.180 | 0.14 | 0.818 |
| C (caption + CLIP, CE) | 0.399 ± 0.008 | 0.746 | 0.749 | 0.635 | 0.146 | 0.12 | 0.816 |
| **C + B1** | **0.416 ± 0.019** | 0.751 | 0.754 | 0.636 | 0.206 | 0.18 | 0.822 |
| M (all inputs, CE) | 0.391 ± 0.011 | 0.760 | 0.758 | 0.657 | 0.077 | 0.13 | 0.830 |
| M + B1 | 0.400 ± 0.007 | 0.747 | 0.750 | 0.623 | 0.152 | 0.15 | 0.823 |
| T without source tags | 0.373 ± 0.008 | 0.754 | 0.760 | 0.649 | 0.017 | 0.01 | 0.826 |
| C without source tags | 0.385 ± 0.018 | 0.741 | 0.746 | 0.628 | 0.098 | 0.07 | 0.814 |

`text-sarcasm` F1 is 0 for every model (one val sample, never predicted).

## 2. Contrasts

| family | contrast | Δ macro-F1 | per seed (1, 2, 42) | 95% CI | p (Holm in family) | group 95% CI |
|---|---|---|---|---|---|---|
| confirmation (§8.4) | C + B1 − C | +0.017 | +0.037 −0.002 +0.015 | [−0.006, +0.041] | 0.16 | [−0.012, +0.040] |
| transfer (§10) | T + B1 − T | **+0.031** | +0.014 +0.051 +0.028 | **[+0.008, +0.056]** | **0.002** | [−0.001, +0.069] |
| transfer (§10) | M + B1 − M | +0.009 | +0.002 +0.001 +0.022 | [−0.016, +0.032] | 0.51 | [−0.026, +0.041] |
| tag retraining (§9) | T no tags − T | −0.008 | −0.023 −0.004 +0.004 | [−0.025, +0.010] | 0.55 | [−0.031, +0.020] |
| tag retraining (§9) | C no tags − C | −0.014 | −0.021 −0.035 +0.014 | [−0.039, +0.007] | 0.55 | [−0.036, +0.008] |
| *exploratory* | C + B1 − T + B1 | +0.005 | +0.036 −0.022 +0.000 | [−0.021, +0.030] | 0.72 | [−0.025, +0.046] |

**B1 confirmation.** On C, B1 gains +0.017 macro-F1 on full train, consistent with the screen (+0.008 out-of-fold) and above the +1-point practical target, but one seed is flat and the interval includes zero: **inconclusive**, with the expected tradeoff (image-sarcasm recall 0.12 → 0.18, F1 0.146 → 0.206; other classes unchanged).

**Transfer.** B1 helps caption-only T clearly: +0.031, positive on all seeds, the only contrast in the study so far whose example-level interval excludes zero after Holm adjustment (the group interval just touches zero). On M the gain is small and uncertain (+0.009).

**Image contribution given B1 (exploratory, not predeclared).** Once both models use B1, adding CLIP to the caption gives +0.005 with seeds of opposite sign. Experiment A's C − T advantage (+0.019 under plain CE) does **not** survive class weighting: much of what looked like an image contribution is reproduced by handling class imbalance in the caption-only model. This is a single post-hoc comparison over three seeds and should be read as a hypothesis for the final benchmark, not a result.

## 3. Source and surface cues (§9)

**Inference edits** on val, OCR and descriptions unchanged (mean over seeds; the edited caption files are in `data/processed/empirical/diagnostics/cue_edits_val.csv`):

| model | source tags removed (193 samples) | all hashtags + signatures (313) | emoji + repeated punctuation (831) |
|---|---|---|---|
| T | flips 24% of edited, ΔF1 −0.009 | 20%, −0.010 | 12%, −0.035 |
| C | 23%, −0.008 | 17%, −0.008 | 12%, −0.020 |
| M | 26%, −0.008 | 20%, −0.008 | 12%, −0.017 |
| C + B1 | 19%, −0.010 | 15%, −0.008 | 10%, −0.021 |

Every model changes about a quarter of its predictions on tagged captions when the tag is removed, so the tag is used; the overall score cost is small (about −0.01) because only 14% of val is tagged. Emoji and punctuation edits touch most captions and cost the most overall (−0.02 to −0.035), largest for caption-only T.

**Retraining without tags** costs −0.008 (T) and −0.014 (C), neither distinguishable from zero; on the tagged subset, C without tags drops from 0.403 to 0.362 (193 samples, descriptive). Removing tags at training time does not improve official-split scores, as the plan anticipated: the tag is a useful in-distribution cue.

**Residual source information.** Removing the caption tag does not remove the source: 30% of tagged train captions and 17% of tagged val captions have a source marker ("interpool", "voz", "WMO"…) in their OCR text, which M reads, and logos may reach CLIP. A fixed sample of 40 tagged val samples for a manual logo check is in `data/processed/empirical/diagnostics/source_logo_audit.csv` (not yet reviewed). No model here should be described as source-free.

## 4. Behaviour of the selected model (§10)

C + B1 relies on the image about as much as C: replacing pixels flips 22% of predictions (seeds 18–28%) vs 20% for C, again most for `image-sarcasm` (31%). Its gain over C comes from minority recall (image recall 0.12 → 0.18), not from higher precision, and not from greater sensitivity to images. On the fixed subsets (descriptive), the largest improvements of C + B1 and T + B1 over their CE versions are on short captions and on the 56 val samples whose caption recurs with different labels.

## 5. Status and next step

- Plan §8.4 confirmation run: done (B1 provisional until B4/B5 are screened or formally dropped).
- Plan §9: inference edits and tag retraining done; manual logo audit pending (`source_logo_audit.csv`).
- Plan §10: transfer done. No further intervention is proposed: the diagnostics do not identify a specific unresolved failure that a single new factor would test.

**Phase 4 (final benchmark)** needs a frozen finalist list before private test is opened once. Based on the predeclared selection, the natural list is C + B1 (selected candidate) with C and T (CE baselines); T + B1 should be included because the exploratory result questions whether the image adds anything once imbalance is handled. B4/B5 must be either screened or formally dropped before freezing.
