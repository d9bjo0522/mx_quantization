#!/bin/bash

# Add the absolute path to microxscaling directory
export PYTHONPATH=/home/tttpd9bjo/mx_quantization/microxscaling:$PYTHONPATH

# Rest of your script
# deit-tiny
# python main.py \
#     --eval \
#     --resume /work/tttpd9bjo/deit/deit_tiny/pretrained_models/deit_tiny_patch16_224-a1311bcf.pth \
#     --model deit_tiny_patch16_224 \
#     --data-path /work/tttpd9bjo/ImageNet \
#     --mx-quant \
#     --top_k \
#     --k 120 \
#     --ex_pred

## deit-small
# python main.py \
#     --eval \
#     --resume /work/tttpd9bjo/deit/deit_small/pretrained_models/deit_small_patch16_224-cd65a155.pth \
#     --model deit_small_patch16_224 \
#     --data-path /work/tttpd9bjo/ImageNet \
#     --mx-quant \
#     --top_k \
#     --k 90 \
#     --ex_pred

## deit-base
python main.py \
    --eval \
    --resume /work/tttpd9bjo/deit/deit_base/pretrained_models/deit_base_patch16_224-b5f2ef4d.pth \
    --model deit_base_patch16_224 \
    --data-path /work/tttpd9bjo/ImageNet \
    --mx-quant \
    --top_k \
    --k 20 \
    --ex_pred \
    --pred_mode "ex_pred"