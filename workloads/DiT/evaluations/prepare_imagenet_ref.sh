#!/bin/bash

current_dir=$(pwd)
ORIGINAL_DIR="/work/tttpd9bjo/ImageNet/val"
NEW_DIR="/work/tttpd9bjo/diffusion/DiT/DiT-XL-2-256x256/evaluation/ImageNet_val"
NUM=${NUM:-1}  # Default to 10 if NUM is not set

rm -rf "$NEW_DIR"
mkdir -p "$NEW_DIR"
cd "$ORIGINAL_DIR"


for class_dir in */; do
    # Remove trailing slash
    class_name="${class_dir%/}"
    src_dir="$ORIGINAL_DIR/$class_name"

    # Copy up to $NUM images
    find "$src_dir" -maxdepth 1 -type f | head -n "$NUM" | while read img; do
        cp "$img" "$NEW_DIR/"
    done
done

cd "$current_dir"