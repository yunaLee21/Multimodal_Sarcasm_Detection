# Empirical study and targeted improvement of Vietnamese multimodal sarcasm detection

Status: proposed protocol, 11 November 2026. This document plans future work; it does not claim that the experiments below have run. Existing results remain historical evidence. This plan replaces the CoT direction as the proposed main study, while preserving its code and reports.

## 1. Objective and contribution

Investigate which factors explain model performance and input dependence on ViMMSD, then test targeted interventions to improve generalization and minority-class recognition.

The project has not established that performance cannot improve. Earlier fusion, retrieval, stacking, and teacher experiments motivate a more controlled study. Successful improvements and well-supported negative results both contribute, but the study will actively pursue improvements.

Working title: **Understanding and Improving Vietnamese Multimodal Sarcasm Detection: Modality Contributions, Class Imbalance, and Surface Cues**.

Expected contributions:

1. A reproducible comparison of caption, image, OCR, and description information under matched settings.
2. Class-specific evidence about model dependence on image information and source/surface cues.
3. Controlled evaluation of imbalance interventions, including label-preserving augmentation.
4. An improved model if supported by held-out results, with a clear account of benefits, tradeoffs, and uncertainty.

## 2. Research questions and hypotheses

| ID | Research question | Testable expectation, not a conclusion |
|---|---|---|
| RQ1 | Which modalities improve performance, and for which classes/subsets? | Image-derived information may help selected cases even when overall gains are small. |
| RQ2 | Which input sources drive predictions? | Caption influence may dominate overall; image influence may differ substantially by class. |
| RQ3 | How do imbalance and scarce minority examples affect learning? | Weighting, sampling, or validated augmentation may improve minority recall, possibly at a precision cost. |
| RQ4 | How much do source and surface cues affect performance and robustness? | Source-tag removal may change predictions; retraining may reduce dependence without necessarily improving official-split scores. |
| RQ5 | Do targeted interventions improve both predictive performance and the diagnosed behavior? | An intervention may improve one dimension without improving the other; report both independently. |

Do not equate sensitivity with reasoning, or a correlation with proof of shortcut learning. A source cue can be useful in-distribution and brittle under distribution change.

## 3. Existing evidence and work to reuse

Sources: `reports/progress_report.md`, `reports/experiment_report.md`, `reports/results.md`, and `reports/dataset_report.md`.

| Existing work | Reuse | Limitation |
|---|---|---|
| Phase 4 baselines, 3 seeds | Historical performance and candidate configurations | Audit exact data, preprocessing, budgets, and selection metrics before combining results. |
| Flat/hierarchical/cascade runs | Historical architectural comparison | These runs are not full teacher distillation. |
| Joint image/OCR/description swapping | Reference diagnostic implementation | Add separate interventions and class/subset breakdowns. |
| Source-tag removal and retraining | Existing controlled evidence | OCR and image logos can retain source information. |
| kNN and stacking | Secondary negative findings | No need to rerun unless the protocol changes materially. |
| Original versus denoised training | Secondary data-quality experiment | Mixed results; do not treat cleaning as established improvement. |
| Teacher generation/rationale evaluation | Brief motivation/history | M1–M3 stopped; do not reopen as core work. |

The reported Phase 4 best private-test macro-F1 is 0.402 ± 0.015 for regularized weighted concat. This is historical, not an untouched-test estimate for a newly selected method. Preserve the distinction between four-class macro-F1 and macro-F1 excluding `text-sarcasm`.

## 4. Data and evaluation protocol

### 4.1 Official splits and prior exposure

Use official train (10,805), public validation (1,413), and private test (1,504), with reference labels as supported by the current loader. Record dataset checksums and class counts after any filtering.

Private-test results have already been viewed for several methods. Disclose this explicitly. Freeze the new protocol and final candidates before any new private evaluation; call it a fixed final benchmark, not a pristine untouched holdout. Do not tune methods after seeing final results. If stronger confirmatory claims are needed, obtain a genuinely unseen, independently annotated evaluation set as an optional extension.

Retain official labels in the main experiment. Flag ambiguous examples for analysis rather than silently relabeling evaluation data. Training-data denoising is a separately named intervention.

### 4.2 Minority-class development within training

Official validation contains only one `text-sarcasm` example and 28 `image-sarcasm` examples. Use a fixed 3-fold group-aware stratified split within official training to develop imbalance interventions.

Build groups using normalized exact caption matches OR exact image hashes, joined transitively. Use an existing group-aware splitter where feasible; validate class support in every fold and publish the resulting manifests. If grouping makes minority support infeasible, document the constraint and fix a feasible split before training. Do not silently fall back to an ungrouped split.

Also report overlap statistics across official splits. Internal grouping prevents leakage within development folds; it does not repair overlap in the official benchmark.

All learned preprocessing, augmentation generation, sampling weights, and class weights must use only the training portion of each fold. Augmented examples inherit their parent's group. Fit neither normalizers nor augmentation selection rules to held-out fold content. Fixed pretrained OCR/descriptions may be reused, with model and cache provenance recorded.

### 4.3 Selection rules

- Internal development: select imbalance methods using pooled out-of-fold four-class macro-F1, accompanied by per-class precision/recall/F1 and support. Each checkpoint is selected using its fold's validation macro-F1. This is a development estimate, not an unbiased final claim after method selection.
- Official validation: use macro-F1 excluding `text-sarcasm` for checkpoint selection, consistently across the matched main comparison. Report all four classes as well.
- Predeclare the final candidate-selection rule: choose at most one imbalance method from internal development, confirm its overall tradeoff on official validation, then compare the fixed candidate and baselines on the final benchmark. Do not choose augmentation based on the single validation text example.
- Report a minority-focused tradeoff if a method improves text/image recognition while reducing overall macro-F1; do not label it an overall improvement.
- Disable automatic private-test evaluation during development (`train.eval_test=false`) and omit private metrics from development summaries and web views.

## 5. Controlled model family

Start with frozen PhoBERT and CLIP, one flat MLP head, identical preprocessing, fixed split manifests, and seeds `[1, 2, 42]`. Use the same seed set across paired comparisons. Keep learning-rate rules, maximum epochs, early stopping, batch construction, image padding, and checkpoint metric matched wherever the architecture permits.

| ID | Input/model | Purpose |
|---|---|---|
| T | Caption → PhoBERT | Caption baseline |
| I | Image pixels → CLIP | Visual baseline, no OCR or description |
| C | Caption + CLIP, regularized concat | Main improvement backbone |
| O | Caption + OCR, separate PhoBERT streams | OCR contribution |
| D | Caption + description, separate PhoBERT streams | Description contribution |
| OD | Caption + OCR + description | Combined image-derived text |
| M | Caption + OCR + description + CLIP | Full multistream model |

Use plain CE for the primary modality matrix so imbalance treatment does not confound architecture. The regularized concat is the primary intervention backbone chosen in advance from historical evidence; do not select it by looking at new private-test results.

O/D/OD/M share the same projection and interaction rules. Where possible keep projection and head sizes fixed. Report parameter counts and compute because changing streams also changes capacity. Add a parameter-matched caption-only head only if claiming gains are specifically due to information rather than capacity. Separate training ablations estimate usefulness after adaptation; inference ablations estimate reliance of a fixed model.

Partially fine-tuned encoders and cross-attention are optional extensions after the core study, each with matched controls. Do not mix frozen and fine-tuned comparisons as evidence about fusion alone.

## 6. Experiment A: modality contributions (RQ1)

Run T/I/C/O/D/OD/M on the official development protocol, 3 seeds each: **21 training runs**, minus verified compatible existing runs.

Predeclared contrasts: C−T; O−T; D−T; OD−O; OD−D; M−OD. Report four-class macro-F1, three-class macro-F1, weighted-F1, accuracy, each class's precision/recall/F1/support, confusion matrices, trainable parameters, training time, and peak VRAM.

Report performance on fixed raw-data subsets: duplicate captions with differing labels, caption-length bins fixed from train, presence/absence of source tags, OCR presence, and train-overlap status. Always provide support; label small subsets descriptive. Avoid claiming significance from tiny minority subsets.

Output: modality comparison table, per-class plots, and paired error-overlap analysis showing examples where multimodal models correct T and where T corrects them.

## 7. Experiment B: input dependence (RQ2)

For C and M, apply the following to the same validation examples without retraining:

1. Replace image pixels only.
2. Replace OCR only (M).
3. Replace descriptions only (M).
4. Replace pixels + OCR + description consistently from one donor (M).
5. Remove OCR or descriptions separately using the model's missing-stream mechanism (M).

Use 5 fixed donor permutations, shared across models/seeds where applicable, with no self-donors. Apply donors from the full split and report outcomes by original class. Optionally add same-label donors as a separate, clearly labeled control; never combine donor policies in a single result.

Keep caption and IDs fixed. For joint replacement, use all donor image views consistently. Record any caption-dependent OCR cleaning and reproduce the fixed preprocessing policy. Store donor mappings so runs are reproducible.

Measure flip rate, mean total-variation probability shift, original→perturbed prediction transition matrices, confidence changes, and changes relative to original labels. **Original labels need not remain valid after substitution**: such score changes measure disruption, not accuracy on a newly labeled task. Arbitrary substitutions may be implausible and out of distribution. Interpret them alongside natural examples and training ablations.

No new training runs required. Grad-CAM and Integrated Gradients provide optional illustrations, not causal proof. Check attribution convergence before relying on examples.

## 8. Experiment C: imbalance and augmentation (RQ3)

### 8.1 Screening matrix

Use C only initially. Run each setting over the fixed 3 internal folds with seed 42: **6 settings × 3 folds = 18 development runs**.

| ID | Treatment |
|---|---|
| B0 | CE, natural sampling |
| B1 | CE with square-root inverse class weights |
| B2 | CE with square-root inverse-frequency sampling weights |
| B3 | Focal loss, gamma 2, without class weights |
| B4 | Reviewed minority augmentation + CE |
| B5 | The same augmentation + square-root inverse class-weighted CE |

B0 and B1 are mandatory controls. B2 is deliberately less aggressive than uniform class balancing. Implement its exact weighting rule explicitly: the current `weighted` sampler must be audited rather than assumed to match this rule. Do not combine oversampling, weighting, and focal loss in the first screen.

Compare fixed optimizer-update budgets, not just epoch counts: augmentation changes dataset size and sampling changes exposure. For each fold, define the budget from B0 before running alternatives. Match validation intervals and warmup/update counts. Report total draws and unique parents seen. This requires budget control beyond simply reusing epoch settings.

### 8.2 Augmentation policy

Default low-cost method: **reviewed back-translation of minority captions**, using the existing `BackTranslator`, without changing the image. The method's validity is an empirical question, especially for Vietnamese wordplay. Manual or LLM paraphrasing is an optional separately named method with its own provenance, cost, and controls; do not mix it invisibly into B4.

Generate at most 2 candidates per minority parent, retaining at most 1 accepted variant per parent. This cap avoids representing a few examples as thousands of nominally independent samples. Apply identically within every fold and later to the full official train. Cache variants by parent, fold eligibility, generator version, parameters, and seed.

Acceptance criteria, checked against the original image:

- Meaning, sentiment, negation, entities, cultural references, joke mechanism, and image references are preserved.
- Sarcastic intent and the original four-class modality attribution remain defensible.
- No invented content, label names, explanation text, or source markers are introduced.
- The variant differs meaningfully from its parent and is not a duplicate of another training example.
- Reject uncertainty; never retain a variant merely to meet a quota.

Have two reviewers independently assess intent preservation and modality-label preservation where feasible. Resolve disagreement or reject; report agreement, acceptance rate by class, rejection reasons, and review time. Reviewers see the original label but should not see model outcomes. Do not use held-out examples to filter variants. If reliable labels cannot be preserved, omit B4/B5 and report the audit rather than training on questionable synthetic data.

Store `augmentation_manifest.jsonl`: parent ID/group, original and generated caption, method/model/version, generation seed/settings, reviewer verdicts, accepted label, and fold membership. Do not use generated captions as independent evaluation examples.

### 8.3 Image augmentation

Treat light image augmentation as a separate optional experiment: brightness/contrast adjustments first, then mild crops only after a readability audit. Existing 90–100% crops can still remove critical meme text. Avoid horizontal flips, erased text, CutMix, and transformations that change the joke. Compare image augmentation alone before combining it with caption augmentation. Changing image pixels requires a declared policy for cached OCR/descriptions; stale caches must not be presented as matching a changed image.

### 8.4 Confirmation and selection

Take at most the best eligible new treatment from B2–B5, using internal OOF macro-F1 with minority metrics as context. If no new treatment beats B1, retain B1 and report that result. Resolve numerical ties in favor of the simpler, cheaper treatment; inspect uncertainty rather than declaring tiny gains conclusive.

Refit C with CE, B1, and the selected treatment on full official train, using seeds 1/2/42. CE is already in Experiment A: **at most 6 additional runs**. If the selected treatment is B1, only 3 additional runs are needed. Measure both official validation scores and internal minority-class evidence. A second internal seed is optional confirmation before refitting if the screen is unstable; declare those extra runs.

## 9. Experiment D: source and surface cues (RQ4)

Apply inference edits separately to T/C/M: source tags only; all hashtags and signatures; emoji and repeated punctuation. Preserve unedited records and save the edited text. Report affected-subset support and effects both over all examples and over edited examples only.

Then retrain T and C with source tags removed from captions, 3 seeds: **6 runs**, subtracting only exact compatible runs. This distinguishes disruption of a model trained with tags from learning without tags. Use the same CE control and budget as Experiment A. Do not change OCR at the same time.

Audit a fixed sample of OCR/source logos to explain residual source information. A caption edit does not remove the source from the image. Do not claim the model is source-free. Length-only and source-tag-only classifiers are inexpensive optional controls, trained on train and evaluated under the same development boundaries.

## 10. Experiment E: targeted improvement and behavior (RQ5)

Compare the selected C intervention against C-CE and C-weighted on original validation inputs, image replacements, cue removal, and fixed natural subsets. Ask whether gains arise from minority recall, improved precision, greater use of useful images, or increased dependence on other cues.

As a limited transfer test, apply the selected imbalance treatment to T and M, 3 seeds each: **at most 6 additional runs**. This tests whether the improvement is specific to C. If no new treatment survives screening, transfer B1 only if its comparison is still missing and useful.

Only if the diagnostics identify a specific unresolved failure should another intervention be added—for example stronger modality dropout. Predeclare the hypothesis, one changed factor, comparator, and budget in an amendment before running. More image sensitivity is not itself the optimization target.

## 11. Statistical analysis and decision rules

- Save per-example probabilities and predictions, not just aggregate metrics.
- Report mean ± sample standard deviation across training seeds; three seeds describe variability but are weak evidence for fine differences.
- For primary contrasts, use paired bootstrap resampling of evaluation examples with 2,000 replicates, preserving model pairing. Where exact-caption/image groups overlap, also use group-resampled bootstrap as a sensitivity analysis. State which uncertainty each interval captures.
- Report seed-specific contrasts and an interval for the mean contrast across the fixed seeds under sample resampling; do not present it as a full estimate of training randomness.
- For the one-example validation text class, do not use bootstrap intervals as evidence of generalization. Base augmentation development on internal OOF support and disclose its selection bias.
- Keep all variants of a parent in the same group. Do not count augmentation variants, donor permutations, or seeds as independent evaluation examples.
- Name primary contrasts before running; report all planned contrasts. If formal hypothesis tests are used, apply Holm adjustment within each declared family; treat additional subset comparisons as exploratory.
- A claimed overall improvement requires a positive held-out macro-F1 difference with uncertainty reported, consistent direction across seeds where possible, and explicit disclosure of per-class regressions. Otherwise describe evidence as inconclusive or a tradeoff.
- Set a practical target of +1 absolute percentage point in four-class macro-F1 before running; smaller reliable gains may still be reported, but distinguish practical size from statistical uncertainty.

Report weighted-F1 and accuracy as secondary metrics. Do not compare the historical 0.402 four-class score directly with a 0.521 three-class score. Prior correction is excluded from the core matrix; if retained as a secondary result, derive it from development data only and never use private-test label proportions.

## 12. Implementation and artifacts

Existing components to reuse: `dataset.py`, `preprocessing.py`, `augmentation.py`, model/fusion modules, losses, experiment runner, diagnostics, metrics, and tracking.

Work required before the new experiments:

1. Add a split/group manifest builder and validator; current random splitting and stacking folds must not be assumed group-aware.
2. Add augmentation generation, review import, and fold-aware dataset expansion. `BackTranslator` alone is not a complete training augmentation pipeline.
3. Add explicit sampling exponent and fixed optimizer-step budget support; log actual draws, updates, and unique parents.
4. Extend diagnostics for pixel/OCR/description interventions independently, fixed donor manifests, and prediction transition matrices.
5. Add subset analysis and paired/group bootstrap reporting with class support.
6. Create separate configs/plans under `configs/experiments/empirical_*`; do not repurpose historical CoT run names or overwrite checkpoints.
7. Add an empirical report builder or separate report. The existing generated experiment report is centered on CoT gates and should remain historical.

Use focused tests for leakage prevention, parent/fold inheritance, class-weight calculation, budget equivalence, intervention isolation, and paired resampling. Run the relevant existing regression tests after changes.

Proposed artifact layout:

```text
data/processed/empirical/splits/       # fold, group, and checksum manifests
data/processed/empirical/augmentation/# candidates, verdicts, accepted variants
outputs/empirical/<experiment>/<run>/ # config, commit, checkpoint, metrics, probabilities
outputs/empirical/diagnostics/        # donor maps, edits, transitions, subset results
outputs/empirical/statistics/         # paired contrasts and confidence intervals
reports/empirical_analysis.md         # final tables, interpretation, limitations
reports/figures/empirical/            # publication/report figures
```

Large data/checkpoints remain outside git. Each run records commit, environment, dataset/cache/split/augmentation hashes, seed, complete resolved config, parameter count, time, VRAM, and selection rule. Preserve actual preprocessing cache keys. Review generated artifacts for sensitive captions before external sharing.

## 13. Budget, sequence, and gates

Core maximum before compatible-run reuse: **57 training runs** (21 modality + 18 internal screening + up to 6 confirmation + 6 tag retraining + up to 6 transfer). Diagnostics require inference, not training. Optional second screening seeds, cross-attention, image augmentation, or fine-tuning are outside this count. If budget is limited, omit transfer and postpone D/OD ablations, keeping matched T/I/C/M baselines and the imbalance controls.

| Phase | Approximate schedule | Deliverable and gate |
|---|---|---|
| 0 | Week 1, first half | Audit historical runs, freeze protocol, validate groups/support, measure one-run runtime. Stop and repair any leakage or invalid split. |
| 1 | Week 1, second half | Implement missing infrastructure and focused checks; generate/review augmentation. Reject invalid augmentation before training. |
| 2 | Week 2 | Run modality comparisons and internal imbalance screening; run baseline diagnostics in parallel where resources permit. |
| 3 | Week 3 | Confirm selected intervention, train tag-removal controls, run transfer if budget permits, repeat diagnostics. |
| 4 | Week 4 | Freeze finalists; perform fixed final benchmark evaluation; compute uncertainty; inspect examples; write report. |

Schedule is provisional until GPU time and reviewer availability are measured. Estimate compute as measured median run duration × remaining runs plus preprocessing/inference overhead, and publish the estimate before launching the full matrix.

Stop expanding the experiment grid if no predeclared hypothesis motivates the next run. A negative imbalance result does not stop analysis; it becomes evidence about the tested methods. A failed augmentation audit stops synthetic training, not the project.

## 14. Qualitative analysis and annotation audit

Select examples using fixed categories and a fixed random seed: multimodal fixes T; multimodal harms T; minority improvement; minority regression; high sensitivity to donors; sensitivity to source removal. Include counterexamples, not only successful illustrations.

Review about 40–60 examples, balancing categories where support allows. Record whether OCR is correct, the caption alone is sufficient, the image adds necessary information, label attribution is ambiguous, and the suspected failure mechanism. Have a second reviewer check a subset and report disagreement. This is descriptive analysis, not a replacement evaluation set.

Official four-way label boundaries are uncertain. Report ambiguity rather than retroactively changing labels to match model reasoning. Avoid claiming that poor minority performance is caused solely by imbalance when label uncertainty or representation limits may contribute.

## 15. Final report structure and completion criteria

Report outline: motivation and questions; dataset/label audit; controlled protocol; modality results; input-dependence diagnostics; imbalance and augmentation results; source-cue experiments; targeted improvement and tradeoffs; qualitative examples; limitations and conclusion.

Required tables/figures: data and split counts; modality matrix; imbalance matrix; per-class F1 with seed variation; intervention probability shifts and transition matrices; subset support/results; augmentation audit; compute costs; final fixed benchmark contrasts.

The study is complete when:

- Every core research question has results or an explicit, justified omission.
- Training/evaluation boundaries, historical test exposure, and protocol amendments are documented.
- Baselines and improvements use matched controls, with per-class metrics and uncertainty.
- Augmentation has an auditable label-preservation review, or a documented rejection decision.
- The best supported candidate and all negative findings are reported without cherry-picking.
- Claims distinguish predictive gains, input dependence, and evidence about generalization.

The final conclusion may be an improved model, a minority-performance tradeoff, or no reliable improvement under the tested interventions. In each case, the contribution is a reproducible explanation of what was tested, what changed, and what remains unresolved.
