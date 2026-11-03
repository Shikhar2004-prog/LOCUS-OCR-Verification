#!/bin/bash
LOCUS_ROOT="${LOCUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# IIIT5K re-run: every verifier now receives the claim in the protocol's canonical form
# (alphanumeric only), matching the case-insensitive alphanumeric evaluation protocol.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate ml_env
cd ${LOCUS_ROOT}
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONPATH=${LOCUS_ROOT}/pydeps CUDA_VISIBLE_DEVICES=$1
L=logs/pipe6; mkdir -p $L
step() { local n=$1; shift; [ -f $L/$n.done ] && return; echo "[$(date +%H:%M)] START $n"
  "$@" > $L/$n.log 2>&1 && touch $L/$n.done && echo "[$(date +%H:%M)] OK $n" || echo "[$(date +%H:%M)] FAIL $n"; }
step x_tess   python code/locus2_features.py IIIT5K tess
step x_pre    python code/locus2_features.py IIIT5K pretrain
step x_trocr  python code/locus2_features.py IIIT5K trocr
step c_tess   python code/features.py IIIT5K tess
step c_trocr  python code/features.py IIIT5K trocr
step vlm      python code/vlm_judge.py IIIT5K
step external python code/eval_external.py IIIT5K
step main     python code/locus.py main IIIT5K
step ablate   python code/locus.py ablate IIIT5K
step single   python code/locus.py ablate_single IIIT5K
step xeng     python code/locus.py xengine IIIT5K
step xds_src  python code/locus.py xdataset IIIT5K
while [ ! -f logs/pipe5/main.done ]; do sleep 30; done   # SROIE as a source needs its new labels
for s in FUNSD CORD IAM SROIE; do XDS_TARGETS=IIIT5K step xds_to_$s python code/locus.py xdataset $s; done
echo "[$(date +%H:%M)] lane IIIT5K finished"
