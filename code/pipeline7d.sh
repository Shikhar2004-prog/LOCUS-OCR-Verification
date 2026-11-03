#!/bin/bash
LOCUS_ROOT="${LOCUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# IAM rendering variants on GPUs with free memory. usage: pipeline7d.sh <gpu> <variant...>
source ~/miniconda3/etc/profile.d/conda.sh; conda activate ml_env; cd ${LOCUS_ROOT}
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONPATH=${LOCUS_ROOT}/pydeps CUDA_VISIBLE_DEVICES=$1; shift
L=logs/pipe7; F=/usr/share/fonts
declare -A FONT=( [sans]=$F/truetype/liberation/LiberationSans-Regular.ttf [serif]=$F/truetype/liberation/LiberationSerif-Regular.ttf
                  [mono]=$F/truetype/liberation/LiberationMono-Regular.ttf [dejavu]=$F/truetype/dejavu/DejaVuSans.ttf
                  [comic]=$F/opentype/comic-neue/ComicNeue-Regular.otf )
for v in "$@"; do
  [ -f $L/IAM_$v.done ] && continue
  case $v in sans0.6|sans0.9) f=sans; sc=${v#sans};; *) f=$v; sc=0.75;; esac
  echo "[$(date +%H:%M)] START IAM_$v"
  SUBSET=1000 RENDER_FONT=${FONT[$f]} RENDER_SCALE=$sc TAG=_r$v python code/locus2_features.py IAM trocr > $L/IAM_$v.log 2>&1 \
    && touch $L/IAM_$v.done && echo "[$(date +%H:%M)] OK IAM_$v" || echo "[$(date +%H:%M)] FAIL IAM_$v"
done
