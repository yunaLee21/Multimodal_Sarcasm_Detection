# Run plan: EACL 2027 SRW (local GPU server, uncommitted)

Date written: 2026-11-23 · Branch: `thanhtruc-server` · **Rule: nothing in this plan gets committed.** New scripts,
configs and doc edits stay in the working tree; outputs go to `data/server/outputs/` (git-ignored, symlink to
`/data/Phat/vimmsd_outputs/`).

Source: the "ViMMSD → EACL 2027 SRW short paper" plan. This file turns it into concrete runs on this machine.

## 0. Facts this plan depends on (checked 2026-11-23)

| Fact | Where | Consequence |
|---|---|---|
| `data.split=official` means train = full train, **val = public test**, test = private test | `src/vimmsd/data/dataset.py:396` | The 9 existing checkpoints were selected on the public test. State this in the paper. |
| 9 checkpoints with `best.pt` exist | `p11_controlled/ca_{ce,la}_s{42,1,2}/`, `p11_caption_la_s{42,1,2}/` | WS-A needs no retraining. |
| Caption-only runs use `model.modality=text` | `configs/experiments/phase11_image_contribution.yaml` | No image encoder is built, so the swap/mask conditions must leave predictions unchanged. This is the sanity check. |
| Image goes through the frozen CLIP encoder at run time: `image_encoder(batch["pixel_values"])` | `src/vimmsd/models/classifier.py:84` | Masking with the "mean embedding" means a forward hook on `model.image_encoder` that returns the train-mean output. Grey image = constant `pixel_values`. |
| Swap helpers exist: `derangement`, `swap_image_views`, `compare_predictions`, `load_trained_run` | `src/vimmsd/analysis/shortcut_check.py` | Reuse them. `swap_image_views` also swaps OCR/description, which are off in Phase 11 (`use_ocr=false`, `use_description=false`). |
| 3-fold group manifest exists | `data/processed/empirical/splits/folds_k3_s42.json` (`data.fold.manifest`, `data.fold.index`) | WS-B uses it as is. |
| Phase 11 matched overrides | `p11_controlled/resolved_configs/ca_la_s42.overrides` | WS-B copies them verbatim and only changes `loss.tau`, the fold, the seed and the output dir. |
| `/dev/shm` is 64 MB | memory note | Every run keeps `train.num_workers=0`. |
| Plan runs can rewrite `reports/experiment_report.md` | memory note | Keep `refresh_report: false`; run `git status` after each run and `git checkout` that file if it changed. |
| No existing cross-attn checkpoints trained on train folds | — | The "out-of-fold train subset" in WS-A only becomes possible after WS-B's τ = 1.0 fold runs (§3). |

## 1. Day 0 (23–24 Nov): cleanup and freeze, no commit

| # | Task | File | Who |
|---|---|---|---|
| 0.1 | Fill the test macro-F1 for `p11_caption_la_s*` (0.3838 / 0.3956 / 0.3892). Re-derive the values from `test_predictions.csv` first, then edit. | `reports/results/results.md:186-188` | Claude |
| 0.2 | Remove the duplicated "8. Still unavailable" section (lines 228 and 301) and renumber §8–9. | `reports/results/phase11_la_controlled.md` | Claude |
| 0.3 | Fix the stale `01_caption_ai_pass1.csv` reference and the truncated last section. | `reports/results/phase11_label_audit/README.md` | Claude |
| 0.4 | Copy `analyze_caption_vs_xattn.py`, `analyze_label_audit.py`, `analyze.py` and the JSON/CSV summaries from `p11_controlled/` into `scripts/analysis/p11/` and `reports/results/p11_controlled/`. **Copy only; the commit waits for your go-ahead.** | — | Claude |
| 0.5 | Write the one-page `reports/plan/protocol.md` with the rules in §2 of the source plan. Team sign-off. | — | Claude drafts, team signs off |
| 0.6 | Snapshot the state: `git rev-parse HEAD`, `git status --porcelain`, `pip freeze` → `data/server/outputs/p12_srw/_launch/`. | — | Claude |

Frozen output root for everything new: `data/server/outputs/p12_srw/`.

## 2. WS-A (24–28 Nov): image-dependence diagnostics, inference only

### 2.1 Code (new, uncommitted)

`scripts/diagnose_image_dependence.py`

```
python scripts/diagnose_image_dependence.py \
  --runs p11_controlled/ca_ce_s42 p11_controlled/ca_ce_s1 p11_controlled/ca_ce_s2 \
         p11_controlled/ca_la_s42 p11_controlled/ca_la_s1 p11_controlled/ca_la_s2 \
         p11_caption_la_s42 p11_caption_la_s1 p11_caption_la_s2 \
  --split test --conditions original swap_random swap_same_label swap_diff_label mask_mean mask_grey \
  --subsets reports/results/phase11_label_audit/coordinator_key.csv \
  --swap-seeds 0 1 2 4 --out data/server/outputs/p12_srw/image_dependence
```

- Load each run with `load_trained_run`. Build the private-test dataset from the run's own `config.json`.
- Conditions (caption fixed):
  - `original`
  - `swap_random`: derangement over the 1,504 test samples. **Repeat with 5 swap seeds**, so the swap noise is reported separately from the model-seed noise.
  - `swap_same_label` / `swap_diff_label`: a derangement within / across gold-label groups. With 7 `text` samples, a same-label derangement is still possible, but say so in the limitations.
  - ~~`swap_same_source` / `swap_diff_source`~~: **dropped 2026-11-23.** Only 229 of 1,504 private-test captions carry a
    source tag (`SOURCE_TAG_RE`), mostly `#interpool` and `#voz_f33`, so the comparison would cover a small, skewed slice.
  - **Audit subsets (as computed 2026-11-23, by rule):** both humans 14, all four raters 11, either human 22, plus
    `audit_prior11`, the 11-id set in `image_needed_human_model_comparison.csv`. The source plan's "11 / 16" does not
    match: 16 is not reproducible from any rule. **Decision 2026-11-23: the 14 both-humans set is the main
    image-needed subset.** 11 (all four raters) and 22 (either human) are sensitivity checks.
  - `mask_mean`: forward hook on `model.image_encoder` that returns the mean image-encoder output over the **train** split. Compute it once and cache it at `p12_srw/image_dependence/train_mean_image_emb.pt`. Train images only, never test.
  - `mask_grey`: every image replaced by a 224×224 RGB (128,128,128) image passed through the run's own `image_processor`.
- Per sample, save: id, gold, pred and the 4-class probabilities under each condition →
  `<out>/<run>/<condition>[_seedK].csv`.
- Summary (`<out>/summary.json`, `<out>/table_image_dependence.csv`):
  - Δ P(gold) mean with a bootstrap CI (B = 10,000, the same resampled ids for every model).
  - Flip rate, plus a flip-direction matrix (orig pred → new pred), split out by gold class.
  - Per-class P/R/F1 and the 4-class macro-F1 under each condition.
  - Everything on three sets: the full private test, the audit "image-needed, both humans agree" set (11) and the
    "image-needed, looser" set (16). The ids come from `coordinator_key.csv` + `0{2,3}_*_person_{1,2}.csv`.
    Check that both counts reproduce 11 and 16 before running.
- Figure: `<out>/fig_flip_direction.pdf`, flip direction by gold class for xattn-LA vs caption-LA under `swap_random` and `mask_mean`.

`tests/test_image_dependence.py`
- A caption-only (`modality=text`) tiny model has identical probabilities under every condition (a hard `allclose` assert).
- The derangement has no fixed points, and the same/diff-label swaps respect the label constraint.
- The mean-embedding hook changes the image-encoder output and nothing else.

### 2.2 Run (Claude runs it; light inference)

1. `pytest tests/test_image_dependence.py tests/test_shortcut_check.py -q`
2. Smoke test: 1 run × `original` only. Its predictions must match `test_predictions.csv` exactly. If they don't, stop; the pipeline is wrong.
3. Full: 9 runs × (1 original + 5 random-swap seeds + 2 controlled swaps + 2 masks) ≈ 90 passes over 1,504 samples. Expect well under an hour on the RTX 5070.
4. Check `nvidia-smi` first. Something was holding 4.6 GB at 2026-11-23 with no compute process listed.

### 2.3 Gate A (28 Nov)

- Is xattn's flip rate under swap/mask on the 11/16 image-needed set above its swap-seed spread and above the full-test rate?
  - **Yes** → claim 2 as written.
  - **No** → negative-finding framing.
- In both cases caption-only must show exactly 0 flips. If it doesn't, the result is invalid until fixed.
- 11 samples × 3 seeds is tiny. Report counts, not percentages, with an exact binomial/permutation p-value. No CI theatre.

## 3. WS-B (26 Nov – 1 Dec): LA τ sweep, optional

### 3.1 Screen: 4 τ × 3 folds × seed 42 = 12 runs

**Status 2026-11-23: ready to launch.**
- Plan: `configs/experiments/p12_tau_screen.yaml`; selection: `scripts/select_tau.py`, both uncommitted.
- The dry run lists the 12 runs, `tau{025,050,075,100}_f{0,1,2}_s42` in `data/server/outputs/p12_srw/`.
- Compared with `p11_controlled/ca_la_s42`, the resolved config differs only in `data.fold.*`, `train.eval_test`
  (true → false), the output dir and the name.
- The fold manifest passes its checksum check: folds of 7,203–7,204 train / 3,601–3,602 held-out samples.
- `select_tau.py` was tested on synthetic OOF predictions.
- **Prior shift caveat:** the held-out folds are about 56% `not` / 39% `multi`, while the private test is 30% / 66%.
  LA corrects using the train class balance, so the τ that is best out-of-fold may not be best on the test. Put this in the limitations.
- Runs append rows to `reports/results/results.md` (uncommitted, like everything else here).

New uncommitted plan `configs/experiments/p12_tau_screen.yaml`:

```yaml
name: p12_tau_screen
seeds: [42]
overrides:            # = p11_controlled ca_la overrides, minus split/seed/name/output
  - train.lr_head=5e-4
  - train.weight_decay=0.05
  - model.dropout=0.3
  - train.early_stop_patience=5
  - loss.class_weight=null
  - loss.mixup_alpha=0.0
  - loss.name=logit_adjustment
  - data.image_augment=false
  - data.augment.file=null
  - data.sampler=random
  - eval.prior_correction=false
  - train.select_metric=macro_f1
  - train.select_exclude_labels=[]
  - train.num_workers=0
  - train.eval_test=false
  - data.fold.manifest=data/processed/empirical/splits/folds_k3_s42.json
  - paths.server.output_dir=data/server/outputs/p12_srw
evaluate: [val]        # val = held-out fold → OOF predictions
refresh_report: false
runs:                  # τ ∈ {0.25, 0.5, 0.75, 1.0} × fold ∈ {0,1,2}
  - {config: configs/fusion_cross_attn_frozen_logit_adjustment.yaml, name: tau025_f0, overrides: [loss.tau=0.25, data.fold.index=0]}
  # … 12 entries in total
```

- **Before launch:** `--dry-run`, then diff one resolved config against `p11_controlled/resolved_configs/ca_la_s42.json`.
  The only allowed differences are `loss.tau`, `data.fold.*`, `train.eval_test`, `name` and `output_dir`.
  Check that `data.split` with a fold manifest gives val = held-out fold, as in `empirical_C_screen.yaml`.
- Cost: about 12 × 15 min ≈ 3 h. **You launch it** (training for hours):

```
conda activate tlenv-sarcasm && cd ~/Multimodal_Sarcasm_Detection && \
VIMMSD_ENV=server nohup python scripts/run_experiments.py \
  --plan configs/experiments/p12_tau_screen.yaml --gpu-ids 0 \
  > data/server/outputs/p12_srw/tau_screen.log 2>&1 &
```

Check that `run.log` shows epoch 1 within about 5 minutes (the `/dev/shm` hang).

- Selection (Claude): concatenate the 3 OOF folds per τ → 4-class macro-F1 (primary) and 3-class macro-F1.
  Bootstrap over the OOF ids.
- **Bonus:** the τ = 1.0 fold checkpoints are cross-attn + LA models that never saw their held-out fold. Rerun WS-A on the
  OOF train image samples (442) with them. This is the only clean way to get the "out-of-fold train subset" in WS-A.
  Caption-only fold controls would need 3 more short runs (`p12_caption_la_f{0,1,2}`, ~1 h). Add them only if Gate A is positive.

### 3.2 Gate B (1 Dec)

- No τ beats τ = 1.0 by more than one fold-SD on OOF macro-F1 → keep τ = 1.0 and stop. One sentence in the paper. No private-test run.
- Otherwise retrain τ* on the full train with `data.split=official`, seeds 42/1/2 (~45 min, you launch it), into
  `p12_srw/final_tau*_s{42,1,2}`. Score the private test **once**, and log the date and command in `p12_srw/_private_test_log.md`.


**Gate B result (2026-11-23, `p12_srw/tau_screen_summary.{csv,json}`):**

| τ | OOF macro-F1 (4) | (3) | Fold SD | − τ1.0 [95% CI] |
|---|---|---|---|---|
| 0.25 | 0.384 | 0.512 | .014 | +0.004 [−0.007, +0.014] |
| 0.50 | 0.388 | 0.512 | .004 | +0.008 [−0.005, +0.023] |
| 0.75 | 0.396 | 0.512 | .012 | +0.016 [−0.001, +0.037] |
| 1.00 | 0.380 | 0.499 | .009 | — |

- The rule selects **τ = 0.75**: +0.016 > the τ = 1.0 fold SD of 0.009.
- The paired CI only just includes 0.
- Part of the 4-class gain is `text` F1 (0.047 vs 0.021, about 100 samples OOF). On 3-class macro-F1 the gain is +0.013.
- Report this as a weak preference for τ < 1.0.

Final plan `configs/experiments/p12_tau_final.yaml`:
- τ = 0.75 and τ = 1.0 (a same-code rerun of `ca_la`) × seeds 42 / 1 / 2, on the full official train; 6 runs, about 1.5 h.
- Compared with `p11_controlled/ca_la_s*`, the resolved config differs only in `loss.tau`, the name and the output dir.
- The `final_tau100` runs should reproduce `ca_la_s*`. Any difference there is code drift and must be reported.

**Final result (2026-11-23, private test scored once; `scripts/compare_tau_final.py` →
`p12_srw/tau_final_comparison_test.json`):**

- Reproduction: `final_tau100_s{42,1,2}` match `ca_la_s{42,1,2}` exactly (0 / 1,504 mismatches, max |Δp| = 0). No code drift.

| τ | Macro-F1 (4), seeds 42 / 1 / 2 | Mean ± SD | Macro-F1 (3) | Accuracy | F1 image | F1 text |
|---|---|---|---|---|---|---|
| 0.75 | 0.434 / 0.385 / 0.375 | 0.398 ± 0.032 | 0.518 ± 0.020 | 0.731 | 0.100 | 0.040 |
| 1.00 | 0.414 / 0.410 / 0.389 | 0.404 ± 0.013 | 0.539 ± 0.018 | 0.755 | 0.126 | 0.000 |

- Paired bootstrap of the seed-averaged difference (τ0.75 − τ1.0, B = 10,000):
  - 4-class: −0.006 [−0.022, +0.010];
  - 3-class: −0.021 [−0.039, −0.004].
- **The OOF gain for τ = 0.75 does not transfer to the private test.** The 4-class difference is null and noisier across seeds. The 3-class result is worse. The only gain is one seed's `text` F1 (0.12, n = 7).
- Paper: keep τ = 1.0 as the main LA setting, and report the τ screen as a negative or prior-shift finding. OOF has 56% `not`, while the private test has 30%, so a τ tuned on train-distribution folds is not expected to transfer. No further τ tuning.

## 4. WS-C (24 Nov – 7 Dec): expanded human audit

| Date | Step | Who |
|---|---|---|
| 24–25 Nov | Guideline v2: `image_contribution` = NA unless the pair is judged sarcastic, plus 3–4 worked examples. Needs team approval (decision 3). | Team; Claude drafts |
| 25 Nov | Sampler `scripts/sample_audit_round2.py --seed 42`: 60 stratified (by gold label, images oversampled) + 60 purposive. Purposive = xattn-LA vs caption-LA disagreements (majority over 3 seeds) and WS-A flip cases. **Exclude the 40 pilot ids.** The purposive half needs WS-A, so run the stratified half on 25 Nov and the purposive half after the WS-A smoke test. | Claude |
| 26 Nov | Forms: reuse the pilot format (`01_caption`, `02_image`, `03_pair` per person), blind to labels, predictions and the other rater. No AI drafts. `coordinator_key.csv` holds gold and stratum and stays with the coordinator only. Output: `reports/results/phase11_label_audit_r2/` (uncommitted). | Claude |
| 26 Nov – 7 Dec | Rating (caption → image → pair). | 2 humans |
| as returned | `analyze_label_audit.py` extended: raw agreement, κ per pass, κ on image role conditioned on both-sarcastic, adjudication table. Report the pilot and round 2 separately and pooled. | Claude |

**Round 2 (2026-11-23): guideline v2 approved; sample drawn to `reports/results/phase11_label_audit_r2/` (plan: `reports/plan/ws_c_round2.md`).** The sampler is `scripts/sample_audit_round2.py --seed 42`.
Changes from the table above:
- `text` has no items left outside the pilot.
- The stratified half is all 23 remaining `image` + 12 `not` + 25 `multi`, with weights.
- The purposive half is 30 xattn-LA vs caption-LA disagreements + 30 WS-A swap-flip ≥ 0.5 items (`not` / `multi`).
- Forms are split into two batches; batch 1 (stratified, Gate A) is rated first.
- Gate A uses a within-gold-class permutation test on the stratified half only.
- The minimum detectable difference is about 0.2 in flip rate.

## 5. WS-D (5–12 Dec): paper. No runs, only table builders

`scripts/build_srw_tables.py` (uncommitted) reads only frozen outputs:

- **Table 1:** controlled comparison, from `p11_controlled/{per_seed_metrics.csv, paired_analysis.json, caption_vs_xattn_stats.json}`.
- **Table 2:** image dependence, from `p12_srw/image_dependence/table_image_dependence.csv`.
- **Figure 1:** `fig_flip_direction.pdf`.
- **Table 3:** audit agreement, pilot + round 2.

Every number in the paper must come from these files. No hand-typed numbers.

**Built 2026-11-23:** `python scripts/build_srw_tables.py` writes to `reports/paper_srw/`:
- `tables/table{1_controlled,2_image_dependence,3_audit,A1_tau}.{tex,csv}` (A1 is the τ appendix);
- `figures/fig_flip_direction.pdf`;
- `numbers.tex` (150 `\newcommand` macros for numbers quoted in the text) and `numbers.json`;
- `MANIFEST.json` (sha256 of every input).

Table 2 is built directly from `arm_condition.csv`, with no intermediate `table_image_dependence.csv`. Table 3 re-analyses the pilot with
`analyze_audit_round2.py`, so the pilot and round 2 use identical code. Its round-2 cells read "pending" until
`phase11_label_audit_r2/analysis/round2_analysis.json` exists, so rerun the builder after each analysis. Pilot Gate A p-values
in the paper are the within-gold-class ones (LA swap p = .533, other = 0.293). They replace the unstratified .46 / 0.287 in
`ws_a_image_dependence.md`. LaTeX is not installed here; `$` and braces were checked for balance but nothing was compiled.

## 6. Day-by-day on this machine

| Date | GPU | Claude | You / team |
|---|---|---|---|
| 23–24 Nov | idle | Day 0 tasks 0.1–0.6; start `diagnose_image_dependence.py` + tests | Decisions 1–3; guideline v2 |
| 25 Nov | WS-A smoke + full run | Sampler (stratified half); source-tag check | Approve guideline |
| 26 Nov | **τ screen (3 h)** | WS-A summary + figure; purposive audit half; forms | Launch τ screen; send forms |
| 27–28 Nov | — | OOF τ selection; **Gate A** | — |
| 29 Nov – 1 Dec | τ* finals only if Gate B passes | WS-A on OOF train with τ = 1.0 fold ckpts; **Gate B** | Launch finals if needed |
| 2–7 Dec | — | WS-A on round-2 audit ids; κ as ratings arrive | Rating |
| 8–12 Dec | — | Table builders, numbers check | Writing, review |

## 7. Guardrails

- The private test is used for WS-A diagnostics (no selection) and for at most one τ* scoring run. Never for choices.
- Swap/mask results are never used to pick a model or a τ.
- Before each launch: `git status` clean except the untracked plan/script files; record `git rev-parse HEAD` in the output dir.
- After each run: `git diff --stat reports/` and revert any auto-rewrite of `reports/experiment_report.md`.
- Nothing is committed until you say so.
