#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/asap-exp-20261123
trap 'echo $? > logs/baselines.exit' EXIT
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
until test -f logs/setup.exit; do sleep 2; done
test "$(cat logs/setup.exit)" = 0
time .venv/bin/python -m aspect_detection train --task aspect --model baseline --output runs/aspect_baseline
time .venv/bin/python -m aspect_detection train --task sentiment --model baseline --output runs/sentiment_baseline
