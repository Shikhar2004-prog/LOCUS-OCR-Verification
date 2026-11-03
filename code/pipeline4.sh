#!/bin/bash
LOCUS_ROOT="${LOCUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Pull-forward lane: transfer experiments that do not depend on SROIE's char evidence.
# Writes pipeline2 done-markers so the original lanes skip these steps.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate ml_env
cd ${LOCUS_ROOT}
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONPATH=${LOCUS_ROOT}/pydeps CUDA_VISIBLE_DEVICES=2
L=logs/pipe2
run() { local n=$1; shift; [ -f $L/$n.done ] && return; touch $L/$n.done  # claim first
  echo "[$(date +%H:%M)] START $n"; "$@" > $L/$n.log 2>&1 && echo "[$(date +%H:%M)] OK $n" || { rm -f $L/$n.done; echo "[$(date +%H:%M)] FAIL $n"; }; }
run FUNSD_xeng python code/locus.py xengine FUNSD
for ds in FUNSD CORD IAM IIIT5K SROIE; do run ${ds}_xds python code/locus.py xdataset $ds; done
echo "[$(date +%H:%M)] lane pull-forward finished"
