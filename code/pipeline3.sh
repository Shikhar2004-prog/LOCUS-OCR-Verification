#!/bin/bash
LOCUS_ROOT="${LOCUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Extra ablation lanes (run alongside pipeline2's main/transfer lanes). usage: pipeline3.sh <gpu> <ds...>
source ~/miniconda3/etc/profile.d/conda.sh; conda activate ml_env
cd ${LOCUS_ROOT}
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONPATH=${LOCUS_ROOT}/pydeps
export CUDA_VISIBLE_DEVICES=$1; shift
L=logs/pipe3; mkdir -p $L
for ds in "$@"; do
  while [ ! -f logs/pipe2/${ds}_main.done ]; do sleep 30; done
  [ -f $L/${ds}_ablate.done ] && continue
  echo "[$(date +%H:%M)] START ${ds}_ablate"
  python code/locus.py ablate $ds > $L/${ds}_ablate.log 2>&1 && touch $L/${ds}_ablate.done \
    && echo "[$(date +%H:%M)] OK ${ds}_ablate" || echo "[$(date +%H:%M)] FAIL ${ds}_ablate"
done
echo "[$(date +%H:%M)] ablation lane finished"
