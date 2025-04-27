#!/bin/bash

# Add the absolute path to microxscaling directory
export PYTHONPATH=/home/tttpd9bjo/mx_quantization/microxscaling:$PYTHONPATH

# Rest of your script
# deit-tiny
python main.py \
    --eval \
    --resume /work/tttpd9bjo/deit/deit_tiny/pretrained_models/deit_tiny_patch16_224-a1311bcf.pth \
    --model deit_tiny_patch16_224 \
    --data-path /work/tttpd9bjo/ImageNet \
    --quantize \
    --top_k \
    --k 100 \
    --exponent_based_prediction

## deit-small
# python main.py \
#     --eval \
#     --resume /work/tttpd9bjo/deit/deit_small/pretrained_models/deit_small_patch16_224-cd65a155.pth \
#     --model deit_small_patch16_224 \
#     --data-path /work/tttpd9bjo/ImageNet \
#     --quantize \
#     --top_k \
#     --k 80 \
#     --exponent_based_prediction

## deit-base
# python main.py \
#     --eval \
#     --resume /work/tttpd9bjo/deit/deit_base/pretrained_models/deit_base_patch16_224-b5f2ef4d.pth \
#     --model deit_base_patch16_224 \
#     --data-path /work/tttpd9bjo/ImageNet \
#     --quantize \
#     --top_k \
#     --k 60 \
#     --exponent_based_prediction