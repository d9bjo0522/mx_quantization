#!/bin/bash

# Add the absolute path to microxscaling directory
root_dir=/home/tttpd9bjo/mx_quantization
mx_dir=$root_dir/microxscaling
deit_dir=$root_dir/workloads/deit

# unset PYTHONPATH

export PYTHONPATH="$root_dir:$mx_dir:$deit_dir"

anal_model_name="deit_base"
anal_dir="/work/tttpd9bjo/deit/${anal_model_name}/analysis"

# deit-tiny
# python main.py \
#     --eval \
#     --resume /work/tttpd9bjo/deit/deit_tiny/pretrained_models/deit_tiny_patch16_224-a1311bcf.pth \
#     --model deit_tiny_patch16_224 \
#     --data-path /work/tttpd9bjo/ImageNet \
#     --anal-dir $anal_dir \
#     --mx-quant \
#     --top-k \
#     --k 80 \
#     --ex-pred \
#     --pred-mode "ex_pred"

## deit-small
# python main.py \
#     --eval \
#     --resume /work/tttpd9bjo/deit/deit_small/pretrained_models/deit_small_patch16_224-cd65a155.pth \
#     --model deit_small_patch16_224 \
#     --data-path /work/tttpd9bjo/ImageNet \
#     --anal-dir $anal_dir \
#     --mx-quant \
#     --top-k \
#     --k 60 \
#     --ex-pred \
#     --pred-mode "ex_pred"

## deit-base
python main.py \
    --eval \
    --resume /work/tttpd9bjo/deit/deit_base/pretrained_models/deit_base_patch16_224-b5f2ef4d.pth \
    --model deit_base_patch16_224 \
    --data-path /work/tttpd9bjo/ImageNet \
    --anal-dir $anal_dir \
    --batch-size 100 \
    --mx-quant \
    --top-k \
    --k 30 \
    --approx-flag \
    --pred-mode "two_step_leading_ones"