#!/bin/bash
LOCUS_ROOT="${LOCUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Overnight orchestrator for LOCUS v2 (dual-verifier). usage: pipeline2.sh <lane>
source ~/miniconda3/etc/profile.d/conda.sh; conda activate ml_env
cd ${LOCUS_ROOT}
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONPATH=${LOCUS_ROOT}/pydeps
D=data; L=logs/pipe2; mkdir -p $L
waitf() { while [ ! -f "$1" ]; do sleep 30; done; }
step() {
  local n=$1; shift
  [ -f $L/$n.done ] && return
  echo "[$(date +%H:%M)] START $n"
  "$@" > $L/$n.log 2>&1 && touch $L/$n.done && echo "[$(date +%H:%M)] OK $n" || echo "[$(date +%H:%M)] FAIL $n (see $L/$n.log)"
}
ext() { step x_$1_$2 python code/locus2_features.py $1 $2; }

case $1 in
xa)  export CUDA_VISIBLE_DEVICES=2
     ext FUNSD tess; ext FUNSD pretrain; ext IAM trocr; ext IAM pretrain; ext SROIE tess; ext SROIE pretrain; ext FUNSD trocr ;;
xb)  export CUDA_VISIBLE_DEVICES=3
     ext CORD tess; ext CORD pretrain; ext IIIT5K tess; ext IIIT5K pretrain; ext CORD trocr; ext IIIT5K trocr; ext SROIE trocr ;;
ea)  export CUDA_VISIBLE_DEVICES=2   # experiments: FUNSD, IAM, SROIE
     for ds in FUNSD IAM SROIE; do
       E=tess; [ $ds = IAM ] && E=trocr
       waitf $D/$ds/locus2_$E.pt; waitf $D/$ds/locus2_pretrain.pt; waitf $D/$ds/feat_$E.pt
       step ${ds}_main python code/locus.py main $ds
     done
     for ds in FUNSD IAM SROIE; do step ${ds}_ablate python code/locus.py ablate $ds; done
     for ds in FUNSD SROIE; do waitf $D/$ds/locus2_trocr.pt; step ${ds}_xeng python code/locus.py xengine $ds; done ;;
eb)  export CUDA_VISIBLE_DEVICES=3   # experiments: CORD, IIIT5K + cross-dataset
     for ds in CORD IIIT5K; do
       waitf $D/$ds/locus2_tess.pt; waitf $D/$ds/locus2_pretrain.pt
       step ${ds}_main python code/locus.py main $ds
     done
     for ds in CORD IIIT5K; do step ${ds}_ablate python code/locus.py ablate $ds; done
     for ds in CORD IIIT5K; do waitf $D/$ds/locus2_trocr.pt; step ${ds}_xeng python code/locus.py xengine $ds; done
     for ds in FUNSD IAM SROIE; do waitf $L/${ds}_main.done; done
     for ds in FUNSD CORD SROIE IAM IIIT5K; do step ${ds}_xds python code/locus.py xdataset $ds; done ;;
esac
echo "[$(date +%H:%M)] lane $1 finished"
