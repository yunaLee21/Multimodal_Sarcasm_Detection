# Paper outline: EACL 2027 SRW short paper (draft 1, 2026-11-23)

Uncommitted. Each number below is referred to by its `numbers.tex` macro (`\name`), with the current value in
brackets for reading only. Do not type the values into the paper. Tables and figures come from `reports/paper_srw/`
(`scripts/build_srw_tables.py`). **Check the page limit and anonymity rules in the SRW call for papers**; this outline
assumes 4 content pages plus unlimited references and a Limitations section.

## Working title (pick one)

1. *Does the Image Matter? Image Dependence and Label Reliability in Vietnamese Multimodal Sarcasm Detection*
2. *The Model Looks at the Image, but Not Where People Do: An Audit of ViMMSD*
3. *Small Classes, Big Claims: Evaluating Multimodal Sarcasm Detection on ViMMSD*

Title 2 only works if round 2 confirms the pilot (Gate A negative). Decide after 2 Dec.

## Thesis (one sentence)

On ViMMSD, a cross-attention model measurably uses the image, but this buys little in macro-F1. Where the model relies on the
image does not clearly match where human raters say the image is needed. And the benchmark's rare classes and label
noise make 4-class macro-F1 too unstable to rank such models.

## Contributions (intro, final paragraph)

1. **A controlled comparison** of caption-only and cross-attention models (CE and LA losses, 3 seeds, paired bootstrap).
   The image-model gain over caption-only is small and its CI includes 0 (`\diffLAvCap` `\diffLAvCapCI` [+0.015, −0.004 to +0.032]).
2. **An image-dependence diagnostic** that keeps the caption fixed and swaps or masks only the image representation, with a
   caption-only control that must stay exactly null. Swaps are stable across seeds. Masks are not (`\flipLAMaskMeanSeedMin`–`\flipLAMaskMeanSeedMax`
   [0.07–0.63]), so we recommend swaps.
3. **A two-round human audit** (caption, image, pair) of the private test that measures label reliability and tests
   whether model image dependence lines up with human "image needed" judgments.
4. **A negative result on prior correction:** an LA τ tuned out-of-fold on train does not transfer to the test under label shift
   (appendix).

## Section plan and page budget (about 4 pages)

### 1. Introduction (0.6 p)
- MSD assumes the image carries signal. MMSD2.0 showed text-only shortcuts on English Twitter (Qin et al., 2023).
  ViMMSD (UIT-DSC 2024) labels *which modality* carries the sarcasm (`not`, `text`, `image`, `multi`), so it should be the
  natural testbed for image use.
- The problem: test has 7 `text` and 40 `image` items out of 1,504. Train and test differ in label mix (`multi` 39% vs 66%).
  The only published ViMMSD baseline reports F1 0.41 with an unclear F1 variant (Vu et al., 2025).
- The questions: does the image help (§3)? does the model use it (§4)? does it use it where humans would (§5)?
- Contributions list (above).

### 2. Setup (0.6 p)
- **Data:** ViMMSD train 10,805; public test used for checkpoint selection (**disclose**); private test 1,504 used for
  reporting. The private test has been viewed in earlier project phases, so call it a fixed final benchmark, not an untouched
  holdout. No choice in this paper was made on it; give the one τ scoring as the only exception, logged.
- **Models:** frozen PhoBERT caption encoder and frozen CLIP ViT-B/32 image encoder; cross-attention fusion vs caption-only
  (same head); CE vs logit adjustment (Menon et al., 2021), τ = 1. 3 seeds (42/1/2), same configs, same commit.
- **Statistics:** paired bootstrap over test ids (B = 10,000) on seed-averaged metrics; permutation tests for audit subsets;
  Holm across the four Gate A tests. Seed spread is reported separately from CIs.
- **Metric note:** 4-class macro-F1 gives `text` (n = 7) a quarter of the weight. We also report 3-class macro-F1 (without `text`).

### 3. Does the image help? (0.5 p) → **Table 1**
- Cross-attn + LA beats caption-only by `\diffLAvCap` [+0.015], but the CI includes 0 and seed 2 shows no gain.
- Cross-attn + CE does not beat caption-only (`\diffCEvCap` [0.000]).
- LA vs CE: `\diffLAvCE` `\diffLAvCECI` [+0.014, +0.001 to +0.028]. Under the group bootstrap over duplicate captions it only
  just holds (`\diffLAvCEGroupCI`; the lower bound is +0.00003, which prints as 0.000). Phrase it as "borderline". The gain comes mostly from `multi` F1.
- F1 `text` = 0 for every run. Seed SD (`\mfFourCESD` [0.023]) is as large as the effects being compared.
- Takeaway: on this benchmark, model ranking by 4-class macro-F1 needs seeds plus paired tests.

### 4. Does the model use the image? (1.0 p) → **Table 2, Figure 1**
- Method: caption fixed; the image-encoder output is replaced by (a) a random other test image, (b) a same-gold-label image, (c) a
  different-gold-label image, (d) the train-mean embedding, (e) a grey image. Validity: every checkpoint reproduces its saved
  predictions exactly, and caption-only flips 0 times under all 16 conditions (`\capFlipMax`).
- Results:
  - LA random swap flips `\flipLASwapRandom` `\flipLASwapRandomCI` [0.159] of predictions.
  - Different-label > random > same-label in both flips and Δp(gold).
  - A same-label image *raises* macro-F1 (`\mfOrigLASwapSameLabel` → `\mfCondLASwapSameLabel` [0.404 → 0.412]). The image carries label-correlated signal.
- Out-of-fold train replication (n = `\oofN`): random swap `\oofFlipSwapRandom` [0.205], same ordering.
- Masks remove sarcasm predictions (`\predSarcOrigLAMaskMean` → `\predSarcCondLAMaskMean`) but vary by seed from `\flipLAMaskMeanSeedMin` to
  `\flipLAMaskMeanSeedMax`. They measure how a seed extrapolates to an out-of-distribution input, not dependence. **Recommendation: use swaps.**
- Figure 1 (flip direction): under swap, gold-`not` moves toward sarcasm and gold-`image` moves away. `image`↔`multi`
  flips are rare. The image shifts *whether* a post is sarcastic far more than *which kind* of sarcasm.

### 5. Does it use the image where humans say it is needed? (0.8 p) → **Table 3**
- Audit design: 2 native-speaker raters, blind, three passes (caption alone → image alone → pair, with image role).
  "Image needed" = caption alone not sarcastic, pair sarcastic, image role = needed (both raters).
  - Pilot: 40 items sampled for prediction instability.
  - Round 2: 60 stratified (weighted to the test) + 60 purposive. Gate A uses the stratified half only, with the analysis fixed before
    rating.
- Reliability:
  - pilot κ caption `\pilotKappaCaption` [0.62], image `\pilotKappaImage` [0.64], pair `\pilotKappaPair` [0.68];
  - image role given a sarcastic pair `\pilotKappaImageRole` [0.47];
  - round-2 weighted values;
  - human-vs-gold disagreement on sarcastic/not.
- Gate A: image-needed vs other items, flip rate under LA swap: pilot `\pilotLASwapNeeded` vs `\pilotLASwapOther` [0.34 vs 0.29],
  p = `\pilotLASwapP` [.53]; round 2 [pending].
- **Two versions, decided by round 2 (2 Dec):**
  - *Gate A negative* (current reading): the model uses the image, but not specifically where people say it is needed.
    Its dependence tracks gold class (gold `image` flips more) more than human judgment.
  - *Gate A positive*: dependence concentrates on human image-needed items. This partially validates the image-sarcasm labels.
- Also report how often humans disagree with gold labels (weighted, round 2). This is the "label reliability" half of the title.

### 6. Discussion and recommendations (0.3 p)
- For ViMMSD users:
  - report 3-class alongside 4-class macro-F1;
  - use ≥ 3 seeds with paired tests;
  - disclose which split was used for checkpoint selection;
  - use swap diagnostics with a caption-only null control, not masks.
- For the dataset: the `text` class (7 test items) and the image-role labels need re-annotation or merging. Point to the audit data release.

### Limitations (required section, not counted)
- Small audit (pilot 40, round 2 120). Gate A can only detect large effects (minimum detectable difference about 0.2 in flip rate; state it).
- Frozen encoders and one fusion architecture. Image dependence of fine-tuned or MLLM models may differ.
- Private test viewed before; checkpoints selected on the public test.
- Train/test label shift: LA τ tuned on train folds did not transfer (Appendix A1: `\tauOOFGain` [+0.016] out-of-fold vs
  `\tauPrivDiffFour` `\tauPrivDiffFourCI` [−0.006, −0.022 to +0.010] on the private test). The pilot audit was sampled for instability.
- Two raters, both team members, though blind to labels and predictions.

### Ethics statement
- Social media posts: the audit release contains ids and ratings only, no images or captions. Raters are team members; nobody was paid. No AI
  ratings mixed into human agreement (AI raters reported separately, or omitted).

### Appendix
- A1 τ screen and final (Table A1).
- A2 audit guideline v2 (translated), with worked examples.
- A3 per-seed tables, flip matrices, sensitivity sets (all-four / either-human).

## Related work (folded into §1 and §2 given space; ~8–12 refs)

All entries below need checking against the actual papers (bib keys, venue, year) before use. Starting points are in
`reports/literature_review/README.md`.
- Datasets: MMSD (Cai et al., 2019), MMSD2.0 (Qin et al., 2023), SarcNet (Yue et al., 2024), the ViMMSD challenge and Vu et al. (2025).
- Encoders and loss: PhoBERT (Nguyen & Nguyen, 2020), CLIP (Radford et al., 2021), logit adjustment (Menon et al., 2021).
- Do multimodal models use both modalities: candidates are input-ablation and perceptual-score work, and MICL's spurious-correlation
  finding (Guo et al., 2024). **Pick 2–3 and verify. I have not checked these citations.**
- Evaluation reliability: seed variance and paired bootstrap; small test classes.

## Open items

| Item | Owner | When |
|---|---|---|
| Decisions 1–2 (first author, owners) | team | now |
| Gate A final → choose title and §5 version | Claude runs, team decides | 2 Dec |
| Round-2 reliability numbers → Table 3 | builder rerun | 2 Dec / 8 Dec |
| Check SRW CfP: page limit, anonymity, whether an appendix is allowed | team | before 5 Dec |
| Verify related-work citations | team | 5–8 Dec |
| LaTeX template + compile check of `tables/*.tex` and `numbers.tex` | team (no LaTeX on server) | 5 Dec |
| Draft §2–§4 (results are final) | — | 5–7 Dec |
| Draft §5, §1, abstract (need Gate A) | — | 8–10 Dec |
| Internal review, cut to page limit | team | 11–12 Dec |
