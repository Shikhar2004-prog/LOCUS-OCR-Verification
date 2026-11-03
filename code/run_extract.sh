#!/bin/bash
LOCUS_ROOT="${LOCUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate ml_env
cd ${LOCUS_ROOT}; export CUDA_VISIBLE_DEVICES=2
stream=$1
if [ "$stream" = A ]; then
  for j in "FUNSD tess" "CORD tess" "IIIT5K tess" "FUNSD pretrain" "CORD pretrain"; do set -- $j; python code/features.py $1 $2 > logs/feat_$1_$2.log 2>&1; done
else
  python code/ocr_hyp.py IAM trocrB > logs/hyp_trocrB_IAM.log 2>&1
  for j in "IAM trocr" "FUNSD trocr" "CORD trocr" "IIIT5K trocr" "IAM pretrain" "IIIT5K pretrain"; do set -- $j; python code/features.py $1 $2 > logs/feat_$1_$2.log 2>&1; done
fi
