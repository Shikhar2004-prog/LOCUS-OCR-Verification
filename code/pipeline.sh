#!/bin/bash
LOCUS_ROOT="${LOCUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Overnight orchestrator. Each "lane" runs on one GPU; every step waits for its inputs.
# usage: pipeline.sh <lane>   (run each lane in its own tmux window)
source ~/miniconda3/etc/profile.d/conda.sh; conda activate ml_env
cd ${LOCUS_ROOT}
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
D=data; L=logs/pipe; mkdir -p $L

waitf() { while [ ! -f "$1" ]; do sleep 30; done; }
step() {  # step <name> <cmd...> ; skips if done-marker exists
  local n=$1; shift
  [ -f $L/$n.done ] && return
  echo "[$(date +%H:%M)] START $n"
  "$@" > $L/$n.log 2>&1 && touch $L/$n.done && echo "[$(date +%H:%M)] OK $n" || echo "[$(date +%H:%M)] FAIL $n (see $L/$n.log)"
}

case $1 in
sroie)   # GPU 4 (~10 GB free): SROIE feature extraction + its baselines
  export CUDA_VISIBLE_DEVICES=4
  step sroie_locus_tess  python code/locus_features.py SROIE tess
  step sroie_locus_pre   python code/locus_features.py SROIE pretrain
  step sroie_locus_trocr python code/locus_features.py SROIE trocr
  step sroie_ceg_tess    python code/features.py SROIE tess
  OMP_NUM_THREADS=1 step sroie_lg python code/linegraph_features.py SROIE tess
  ;;
vlm)     # GPU 3 after the running VLM jobs: SROIE judge
  export CUDA_VISIBLE_DEVICES=3
  while pgrep -f jobs_vlm.sh >/dev/null; do sleep 60; done
  step vlm_SROIE python code/vlm_judge.py SROIE
  ;;
exp2)    # GPU 2: experiments for CORD, IIIT5K, SROIE
  export CUDA_VISIBLE_DEVICES=2
  for ds in CORD IIIT5K SROIE; do
    E=$( [ $ds = IAM ] && echo trocr || echo tess )
    waitf $D/$ds/locus_$E.pt; waitf $D/$ds/locus_pretrain.pt; waitf $D/$ds/feat_$E.pt
    step ${ds}_main   python code/locus.py main $ds
    step ${ds}_ablate python code/locus.py ablate $ds
  done
  for ds in CORD IIIT5K SROIE; do waitf $D/$ds/locus_trocr.pt; step ${ds}_xeng python code/locus.py xengine $ds; done
  ;;
exp3)    # GPU 3: experiments for FUNSD, IAM + cross-dataset
  export CUDA_VISIBLE_DEVICES=3
  for ds in FUNSD IAM; do
    E=$( [ $ds = IAM ] && echo trocr || echo tess )
    waitf $D/$ds/locus_$E.pt; waitf $D/$ds/locus_pretrain.pt; waitf $D/$ds/feat_$E.pt
    step ${ds}_main   python code/locus.py main $ds
    step ${ds}_ablate python code/locus.py ablate $ds
  done
  waitf $D/FUNSD/locus_trocr.pt; step FUNSD_xeng python code/locus.py xengine FUNSD
  for ds in CORD IIIT5K SROIE; do waitf $D/$ds/locus_pretrain.pt; done
  for ds in FUNSD CORD SROIE IAM IIIT5K; do step ${ds}_xds python code/locus.py xdataset $ds; done
  ;;
reocr)   # GPU 3: PARSeq (ECCV'22) + Florence-2 (CVPR'24) re-recognition baselines
  export CUDA_VISIBLE_DEVICES=3 PYTHONPATH=${LOCUS_ROOT}/pydeps
  for ds in FUNSD CORD IAM IIIT5K SROIE; do
    waitf $D/$ds/hyp_tess.jsonl
    [ $ds = FUNSD ] || step reocr_parseq_$ds python code/reocr_baselines.py $ds parseq
    step reocr_florence_$ds python code/reocr_baselines.py $ds florence
  done
  ;;
esac
echo "[$(date +%H:%M)] lane $1 finished"
