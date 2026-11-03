#!/bin/bash
LOCUS_ROOT="${LOCUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Re-run every SROIE-dependent experiment after switching SROIE to case-insensitive labels
# (its ground truth is entirely upper-case). usage: pipeline5.sh <lane>
source ~/miniconda3/etc/profile.d/conda.sh; conda activate ml_env
cd ${LOCUS_ROOT}
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONPATH=${LOCUS_ROOT}/pydeps
L=logs/pipe5; mkdir -p $L
step() { local n=$1; shift; [ -f $L/$n.done ] && return; echo "[$(date +%H:%M)] START $n"
  "$@" > $L/$n.log 2>&1 && touch $L/$n.done && echo "[$(date +%H:%M)] OK $n" || echo "[$(date +%H:%M)] FAIL $n"; }
case $1 in
a) export CUDA_VISIBLE_DEVICES=2
   step main   python code/locus.py main SROIE
   step ablate python code/locus.py ablate SROIE
   step single python code/locus.py ablate_single SROIE ;;
b) export CUDA_VISIBLE_DEVICES=3
   step external python code/eval_external.py SROIE
   step xeng     python code/locus.py xengine SROIE
   step xds_src  python code/locus.py xdataset SROIE ;;
c) export CUDA_VISIBLE_DEVICES=3 XDS_TARGETS=SROIE
   for s in FUNSD CORD IAM IIIT5K; do step xds_$s python code/locus.py xdataset $s; done ;;
esac
echo "[$(date +%H:%M)] lane $1 finished"
