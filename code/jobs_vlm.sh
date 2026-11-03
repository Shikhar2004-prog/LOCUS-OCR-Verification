#!/bin/bash
LOCUS_ROOT="${LOCUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate ml_env
cd ${LOCUS_ROOT}; export CUDA_VISIBLE_DEVICES=3
for d in FUNSD CORD IAM IIIT5K; do python code/vlm_judge.py $d > logs/vlm_$d.log 2>&1; done
