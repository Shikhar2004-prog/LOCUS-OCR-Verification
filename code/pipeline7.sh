#!/bin/bash
LOCUS_ROOT="${LOCUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Rendering-sensitivity study: re-extract dual-verifier evidence with different fonts / glyph sizes
# for R(v) on a fixed subset (first 2000 train + 2000 test crops). usage: pipeline7.sh <gpu> <ds...>
source ~/miniconda3/etc/profile.d/conda.sh; conda activate ml_env
cd ${LOCUS_ROOT}
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONPATH=${LOCUS_ROOT}/pydeps CUDA_VISIBLE_DEVICES=$1; shift
L=logs/pipe7; mkdir -p $L
F=/usr/share/fonts
declare -A FONT=( [sans]=$F/truetype/liberation/LiberationSans-Regular.ttf [serif]=$F/truetype/liberation/LiberationSerif-Regular.ttf
                  [mono]=$F/truetype/liberation/LiberationMono-Regular.ttf [dejavu]=$F/truetype/dejavu/DejaVuSans.ttf
                  [comic]=$F/opentype/comic-neue/ComicNeue-Regular.otf )
run() { local n=$1; shift; [ -f $L/$n.done ] && return; echo "[$(date +%H:%M)] START $n"
  "$@" > $L/$n.log 2>&1 && touch $L/$n.done && echo "[$(date +%H:%M)] OK $n" || echo "[$(date +%H:%M)] FAIL $n"; }
for ds in "$@"; do
  E=tess; [ $ds = IAM ] && E=trocr
  for f in sans serif mono dejavu comic; do
    SUBSET=2000 RENDER_FONT=${FONT[$f]} RENDER_SCALE=0.75 TAG=_r$f run ${ds}_$f python code/locus2_features.py $ds $E
  done
  for sc in 0.6 0.9; do
    SUBSET=2000 RENDER_FONT=${FONT[sans]} RENDER_SCALE=$sc TAG=_rsans$sc run ${ds}_sans$sc python code/locus2_features.py $ds $E
  done
done
echo "[$(date +%H:%M)] font lane finished"
