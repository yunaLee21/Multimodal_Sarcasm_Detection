# Reviewing back-translated captions (empirical plan, section 8.2)

Each reviewer fills their own sheet independently: `data/processed/empirical/augmentation/review_<name>.csv`. Do not look at the other reviewer's sheet or at any model output. Open the image (`image_path`) for every row: the original label depends on the image and the caption together.

Each row is one candidate: an automatic Vietnamese → English → Vietnamese rewrite of a minority-class caption. The image stays the same.

## Columns to fill

| column | value | meaning |
|---|---|---|
| `meaning_preserved` | y / n | Same meaning, sentiment, negation, named entities and cultural references; any reference to the image still works. |
| `sarcastic_intent_preserved` | y / n | The joke or sarcasm mechanism survives (wordplay, slang, teencode, tone). A literal rewrite that kills the pun is **n**. |
| `modality_label_still_valid` | y / n | With this image, the original label (`text-sarcasm` or `image-sarcasm`) is still defensible for the new caption. |
| `no_invented_content` | y / n | No added facts, label words, explanations or source tags (page names, hashtags) that the original did not have. |
| `verdict` | accept / reject | **accept** only if all four columns above are y and the text reads like a real caption. Otherwise reject. |
| `reason` | free text | Short reason when rejecting (e.g. "pun lost", "wrong entity", "unnatural"). |
| `minutes` | number | Optional: minutes spent on this row, for the review-time report. |

## Rules

- When unsure, reject. Never accept a candidate to reach a quota.
- The two candidates of the same parent are judged separately; at most one accepted candidate per parent is used.
- A candidate is used only if **both** reviewers accept it with all four criteria = y. Disagreements are rejected.
- Rows are sorted with `text-sarcasm` first. If time is short, finish `text-sarcasm` completely before starting `image-sarcasm`, and tell the team, so the partial review is reported honestly.

When both sheets are done:

```bash
VIMMSD_ENV=server python scripts/augment_minority.py import \
  --reviews reviewer1=data/processed/empirical/augmentation/review_reviewer1.csv \
            reviewer2=data/processed/empirical/augmentation/review_reviewer2.csv
```

If reliable labels cannot be preserved for enough candidates, treatments B4/B5 are dropped and the audit is reported instead (plan section 8.2).
