#!/bin/bash
LOCUS_ROOT="${LOCUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate ml_env; cd ${LOCUS_ROOT}
export CUDA_VISIBLE_DEVICES=$1; shift
for d in "$@"; do python code/run.py main $d > logs/main_$d.log 2>&1; done
