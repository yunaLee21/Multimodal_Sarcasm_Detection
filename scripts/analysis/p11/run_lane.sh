#!/usr/bin/env bash
# usage: run_lane.sh <arm: ce|la>   — runs seeds 42 1 2 sequentially, retries once on failure
cd /home/tltest/Multimodal_Sarcasm_Detection
export VIMMSD_ENV=server
PY=/home/tltest/.conda/envs/tlenv-sarcasm/bin/python
OUT=data/server/outputs/p11_controlled
arm=$1
[ "$arm" = ce ] && CFG=configs/fusion_cross_attn_frozen.yaml || CFG=configs/fusion_cross_attn_frozen_logit_adjustment.yaml
for seed in 42 1 2; do
  name=ca_${arm}_s${seed}
  read -r -a OV < $OUT/resolved_configs/$name.overrides
  for attempt in 1 2; do
    if [ -f $OUT/$name/test_predictions.csv ] && [ -f $OUT/$name/val_predictions.csv ]; then break; fi
    rm -rf $OUT/$name; mkdir -p $OUT/$name
    { $PY scripts/train.py --config $CFG --override "${OV[@]}" --note "Phase11 controlled $arm seed $seed" \
      && $PY scripts/evaluate.py --config $CFG --split test --override "${OV[@]}" \
      && $PY scripts/evaluate.py --config $CFG --split val --override "${OV[@]}"; } > $OUT/$name.log 2>&1 \
      && { mv $OUT/$name.log $OUT/$name/train_eval.log; rm -f $OUT/$name/last.pt; echo "$(date) $name OK"; break; } \
      || echo "$(date) $name FAILED attempt $attempt"
  done
done
