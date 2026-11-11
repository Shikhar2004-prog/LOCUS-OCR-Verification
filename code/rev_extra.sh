#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh; conda activate ml_env; cd /storage/shikhar/ocrv_ext
g=$1; shift
for job in "$@"; do e=${job%:*}; d=${job#*:}; CUDA_VISIBLE_DEVICES=$g python code/revision_exp.py $e $d >> logs/rev/${e}_$d.log 2>&1; done
