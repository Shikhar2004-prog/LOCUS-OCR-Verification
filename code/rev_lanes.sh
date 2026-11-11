#!/bin/bash
# lane A (GPU $1): noise + gsize ; lane B (GPU $2): pseudo + xcal
source ~/miniconda3/etc/profile.d/conda.sh; conda activate ml_env
cd /storage/shikhar/ocrv_ext
laneA(){ for d in IIIT5K FUNSD CORD SROIE IAM; do for e in noise gsize; do CUDA_VISIBLE_DEVICES=$1 python code/revision_exp.py $e $d > logs/rev/${e}_$d.log 2>&1; done; done; }
laneB(){ for d in IIIT5K FUNSD CORD SROIE IAM; do CUDA_VISIBLE_DEVICES=$2 python code/revision_exp.py pseudo $d > logs/rev/pseudo_$d.log 2>&1; done
         for d in IIIT5K FUNSD CORD SROIE; do CUDA_VISIBLE_DEVICES=$2 python code/revision_exp.py xcal $d > logs/rev/xcal_$d.log 2>&1; done; }
laneA $1 & laneB $1 $2 & wait
