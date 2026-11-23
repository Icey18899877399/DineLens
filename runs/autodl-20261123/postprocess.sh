#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/asap-exp-20261123
trap 'echo $? > logs/postprocess.exit' EXIT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=3
for job in baselines bert; do
  until test -f "logs/$job.exit"; do sleep 5; done
  test "$(cat "logs/$job.exit")" = 0
done
.venv/bin/python -m aspect_detection predict --run runs/joint_bert --device cuda \
  --text '菜很好吃，但是等了四十分钟才上菜，前台态度很差。' \
  --evidence --tfidf-run runs/aspect_baseline > runs/bert_evidence_demo.json
if .venv/bin/python -c 'import csv,sys; rows=csv.DictReader(open("annotations/asap_evidence_test.csv", encoding="utf-8-sig", newline="")); sys.exit(not any(row["label"].strip() for row in rows))'; then
  .venv/bin/python -m aspect_detection evidence-evaluate \
    --annotations annotations/asap_evidence_test.csv \
    --tfidf-run runs/aspect_baseline --output runs/evidence_evaluation.json
else
  echo "No human evidence labels are present; recording candidate diagnostics."
fi
.venv/bin/python report_results.py
.venv/bin/python -m pip freeze --local > runs/remote_dependencies.txt
tar -cf results.tar runs logs run_bert.sh run_baselines.sh postprocess.sh report_results.py
sha256sum results.tar > results.tar.sha256
