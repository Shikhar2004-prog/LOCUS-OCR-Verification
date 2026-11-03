#!/bin/bash
LOCUS_ROOT="${LOCUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# IAM font variants on a second GPU, reverse order (meets pipeline7b in the middle)
source ~/miniconda3/etc/profile.d/conda.sh; conda activate ml_env; cd ${LOCUS_ROOT}
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONPATH=${LOCUS_ROOT}/pydeps CUDA_VISIBLE_DEVICES=7
L=logs/pipe7; F=/usr/share/fonts; S=$F/truetype/liberation/LiberationSans-Regular.ttf
run() { local n=$1; shift; [ -f $L/$n.done ] && return; touch $L/$n.claimed; echo "[$(date +%H:%M)] START $n"
  "$@" > $L/$n.log 2>&1 && touch $L/$n.done && echo "[$(date +%H:%M)] OK $n" || echo "[$(date +%H:%M)] FAIL $n"; }
for sc in 0.9 0.6; do SUBSET=1000 RENDER_FONT=$S RENDER_SCALE=$sc TAG=_rsans$sc run IAM_sans$sc python code/locus2_features.py IAM trocr; done
SUBSET=1000 RENDER_FONT=$F/opentype/comic-neue/ComicNeue-Regular.otf TAG=_rcomic run IAM_comic python code/locus2_features.py IAM trocr
SUBSET=1000 RENDER_FONT=$F/truetype/dejavu/DejaVuSans.ttf TAG=_rdejavu run IAM_dejavu python code/locus2_features.py IAM trocr
echo "[$(date +%H:%M)] font lane C finished"
