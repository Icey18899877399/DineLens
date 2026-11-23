#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/asap-exp-20261123
trap 'echo $? > logs/bert.exit' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 RAYON_NUM_THREADS=4
until test -f logs/weights.ready; do sleep 2; done
test "$(cat logs/setup.exit)" = 0
echo "Started at $(date -u +%FT%TZ)"
time .venv/bin/python -m aspect_detection train --task sentiment --model bert \
  --dataset dataset --output runs/joint_bert --checkpoint checkpoints/bert-base-chinese \
  --device cuda --epochs 3 --batch-size 8 --max-length 512 --seed 42
echo "Finished at $(date -u +%FT%TZ)"
