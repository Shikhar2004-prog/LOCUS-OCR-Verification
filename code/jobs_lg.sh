#!/bin/bash
LOCUS_ROOT="${LOCUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# prior-work (line graph) features for the remaining datasets
source ~/miniconda3/etc/profile.d/conda.sh; conda activate ml_env
cd ${LOCUS_ROOT}; export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1
for j in "IIIT5K tess" "IAM trocr"; do
  set -- $j
  timeout 3h python code/linegraph_features.py $1 $2 > logs/lg_$1_$2.log 2>&1
done
