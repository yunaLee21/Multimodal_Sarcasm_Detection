# WS-C round 2: human-audit sample plan

Drafted 2026-11-23. **Guideline v2 approved 2026-11-23. Sample drawn the same day** into `reports/results/phase11_label_audit_r2/`:
`batch{1,2}/` forms, `images/` (120), `coordinator_key.csv` and `README.md` (rater instructions, v2). Uncommitted.
Defaults taken: `khong_chac` on the pair pass also forces `NA`; three worked examples (no `bo_sung` example).

Sampler: `scripts/sample_audit_round2.py --seed 42` (dry run by default; `--write --copy-images` to draw).
The write path was tested in the scratchpad only: 120 items, 120 images, forms in the pilot format.

## 1. What round 2 must answer

1. **Gate A, final (main).** On items *not* chosen for model behaviour, do the ones humans judge "image needed" flip more under
   an image swap than the others? The pilot could not answer this because it was sampled for prediction instability
   (LA swap 0.34 vs 0.29, p = 0.46, n = 14).
2. **Label reliability on the private test.** Agreement per pass (caption, image, pair) and on image role, plus how often
   humans agree with the benchmark gold, on a sample that can be reweighted to the test distribution.
3. **Descriptive.** When the model's prediction depends on the image (cross-attn and caption-only disagree, or a swap
   flips the prediction), do humans say the image is needed?

## 2. Pool

Private test, excluding the 40 pilot ids: **1,464 items: 444 `not`, 23 `image`, 997 `multi`, 0 `text`.** The pilot
took all 7 `text` items, so round 2 has no `text` stratum. `text` agreement comes from the pilot only (n = 7).

## 3. Design: 120 items in two halves

| Half | Stratum | Rule | n | Pool | Weight |
|---|---|---|---|---|---|
| **Stratified** (batch 1) | `strat_image` | all remaining gold `image` | 23 | 23 | 1.0 |
| | `strat_not` | random | 12 | 444 | 37.0 |
| | `strat_multi` | random | 25 | 997 | 39.9 |
| **Purposive** (batch 2) | `disagree_not` / `disagree_multi` | majority vote over 3 seeds: xattn-LA ≠ caption-LA | 15 + 15 | 56 / 134 | — |
| | `flip_not` / `flip_multi` | xattn-LA random-swap flip rate ≥ 0.5 (3 seeds × 5 swaps), not already drawn | 15 + 15 | 46 / 60 | — |

- `image` is oversampled (all 23) because image-needed items come mostly from it: in the pilot, 11 of 17 gold `image` items
  were both-humans image-needed, compared with 3 of 14 `not` and 0 of 2 `multi`.
- Population estimates from the stratified half use the weights. They are never pooled with the purposive half.
- Model signals come from the existing WS-A outputs (`p12_srw/image_dependence/*/probs.npz`). The sampler needs no new
  inference.
- Audit ids `R001`–`R120` are shuffled over all 120 items, and each pass is shuffled separately within a batch, so neither the
  ids nor the order give away the half. `coordinator_key.csv` (gold, stratum, weight, model signals) stays with the
  coordinator.

**Why two batches.** Batch 1 (stratified) carries the Gate A test, so it is rated first and must finish even if
batch 2 slips. Raters are not told how the batches differ.

## 4. Guideline v2 (draft for decision 3)

Changes to the pilot instructions (`phase11_label_audit/README.md`, "Hai người đánh giá độc lập"):

1. **`image_contribution` = `NA` unless the pair is judged sarcastic (`co`).** In the pilot, person 2 gave `can` or `bo_sung`
   on 10 pairs that both raters judged *not* sarcastic. That makes κ on image role uninterpretable.
2. **One test for `can` versus the rest:** "Cover the image and read only the caption. Is the sarcasm still recognisable? No → `can`.
   Yes, but the image makes it clearer → `bo_sung`. Yes, and the image adds nothing → `khong`."
3. **Worked examples from the pilot** (both raters agreed; these items are not in round 2):
   - `A008`: `co` / `can` (the caption is an ordinary question; the answer shown in the image makes it sarcastic);
   - `A033`: `co` / `khong` (the caption alone is sarcastic);
   - `A002`: `khong` → `NA` (news sharing; the pilot's "image adds info" answer is exactly what v2 removes);
   - `bo_sung`: no pilot item had both raters agree on it. The team picks one, or we keep three examples.

Draft wording to insert in the round-2 README (Vietnamese, matching the pilot):

> Ở lượt cả cặp, chỉ điền `image_contribution` khi `sarcasm` = `co`. Nếu `sarcasm` là `khong` hoặc `khong_chac`,
> điền `NA`. Cách phân biệt: che ảnh và chỉ đọc caption. Nếu không còn nhận ra ý châm biếm, điền `can`. Nếu vẫn nhận ra
> nhưng ảnh làm rõ thêm, điền `bo_sung`. Nếu ảnh không thêm gì, điền `khong`.

Other rules are unchanged from the pilot: three passes in order (caption → image → pair); each pass is collected before the next is handed out;
raters are blind to gold, predictions and each other; no AI drafts.

## 5. Analysis, fixed before rating

**Gate A (stratified half only):**
- *Image-needed*, as in the pilot: caption pass ≠ `co`, pair = `co`, `image_contribution` = `can`. The main set is
  both humans, with all-four-raters and either-human as sensitivity checks; AI raters are used only if added later.
- *Outcome:* per-item xattn-LA random-swap flip rate (already computed). Secondary outcomes: CE swap, and the masks
  (reported, but they are seed-unstable).
- *Test:* difference in mean flip rate, image-needed vs other. The p-value comes from a permutation test **permuting
  within gold class**, because image-needed items cluster in gold `image`, which already flips more (OOF 0.27 vs 0.19).
  10,000 permutations; Holm correction across the four arm × condition tests.
- *Expected size and power:* about 15–24 image-needed items, based on the pilot's per-class rates. The SD of the per-item
  flip rate in the drawn half is 0.24. The minimum detectable difference (80% power, α = .05) is about 0.18–0.22,
  against a base rate of about 0.2. **Round 2 can detect "image-needed items flip about twice as often", not a subtle
  effect.** The paper states this.
- *Decision:* a Holm-significant positive difference on LA random swap → the strong form of claim 2. Otherwise the paper
  keeps the weak form ("the model uses the image, but not specifically where people say it should").
- The pilot and round 2 are reported separately. A pooled analysis is labelled as including the instability-sampled pilot.

**Reliability (stratified half, weighted to the test distribution):**
- raw agreement and Cohen's κ per pass;
- κ on image role, conditioned on both raters judging the pair `co`;
- rate of human-pair vs gold disagreement, weighted, with a CI;
- an adjudication table listing every disagreement.

**Purposive half (descriptive only, no tests):**
- rate of image-needed in `disagree_*` and `flip_*` against the weighted stratified rate. This answers the reverse direction: when the image moves the model, do humans say it is needed?

Script: `scripts/analyze_audit_round2.py` (written 2026-11-23; the pilot script is left unchanged). It runs whatever is complete:
a batch is analysed once both humans have returned all three sheets for it, as `batch{1,2}/<pass>_person_{1,2}.csv`.
It writes `phase11_label_audit_r2/analysis/{round2_analysis.json, items.csv, adjudication_batch{1,2}.csv}`. It also checks values,
missing answers and v2 rule violations. Validation:
- On the pilot data it reproduces the WS-A both-humans means (LA swap 0.338, LA mask 0.571, CE 0.233 / 0.357).
- On synthetic round-2 sheets with a planted effect it reports LA swap Holm p = 0.002.
- Unit tests are in `tests/test_audit_round2.py`.
- One known difference from the WS-A pilot numbers: an item a rater left blank is excluded from that set. WS-A counted it as "not needed".

## 6. Timeline

| Date | Step | Who |
|---|---|---|
| 24–25 Nov | Approve guideline v2 and choose the `bo_sung` example. Sign off `protocol.md`. | Team |
| on approval | `--write --copy-images`, write the round-2 README (pilot text + v2 changes), hand out batch 1 caption pass | Claude, then coordinator |
| by 1 Dec | Batch 1 returned (3 passes × 60 items per rater) | 2 humans |
| 1–2 Dec | Gate A final on batch 1 | Claude |
| by 7 Dec | Batch 2 returned | 2 humans |
| 7–8 Dec | Reliability and purposive analyses, adjudication table | Claude, then team |

Rater load: 360 judgements per rater in total, three times the pilot's 120. If time runs short, drop batch 2. Batch 1 alone
still decides Gate A and gives the reliability numbers.

## 7. Decisions needed

1. Approve guideline v2 (§4), including whether `khong_chac` on the pair pass also forces `NA`. The draft says it does.
2. The `bo_sung` worked example: pick one, or use three examples.
3. OK with 120 items in two batches, or cut to the 60-item stratified half only?
4. Keep the round-2 forms and images out of git, as with the pilot images? The default is yes, and nothing is committed anyway.
