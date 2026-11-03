#!/bin/bash
LOCUS_ROOT="${LOCUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# message-passing analysis on all datasets (SROIE/IIIT5K after their re-runs finish)
source ~/miniconda3/etc/profile.d/conda.sh; conda activate ml_env; cd ${LOCUS_ROOT}
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONPATH=${LOCUS_ROOT}/pydeps CUDA_VISIBLE_DEVICES=3
for ds in FUNSD CORD IAM SROIE IIIT5K; do
  [ $ds = SROIE ] && while [ ! -f logs/pipe5/main.done ]; do sleep 30; done
  [ $ds = IIIT5K ] && while [ ! -f logs/pipe6/main.done ]; do sleep 30; done
  echo "[$(date +%H:%M)] START mp_$ds"; python code/locus.py mp_analysis $ds > logs/mp_$ds.log 2>&1 && echo "[$(date +%H:%M)] OK mp_$ds" || echo "[$(date +%H:%M)] FAIL mp_$ds"
done
