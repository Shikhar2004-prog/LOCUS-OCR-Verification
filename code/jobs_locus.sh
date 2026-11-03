#!/bin/bash
LOCUS_ROOT="${LOCUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# LOCUS edit-graph extraction. usage: jobs_locus.sh <gpu> "DS ENGINE" ...
source ~/miniconda3/etc/profile.d/conda.sh; conda activate ml_env
cd ${LOCUS_ROOT}; export CUDA_VISIBLE_DEVICES=$1; shift
for j in "$@"; do
  set -- $j
  python code/locus_features.py $1 $2 > logs/locus_$1_$2.log 2>&1
done
