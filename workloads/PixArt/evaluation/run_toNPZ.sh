# Use environment variables with defaults, or command line arguments
SAMPLE_DIR=${SAMPLE_DIR:-${1:-"/work/tttpd9bjo/MSCOCO/val2017"}}
NPZ_DIR=${NPZ_DIR:-${2:-"/work/tttpd9bjo/diffusion/PixArt/PixArt-XL-2/evaluation/PixArt-XL-2-512x512"}}
FILE_NAME=${FILE_NAME:-${3:-"val2017_512_5000"}}
NUM=${NUM:-${4:-"5000"}}
RESOLUTION=${RESOLUTION:-${5:-"512"}}

python toNPZ.py \
    --sample_dir "$SAMPLE_DIR" \
    --npz_dir "$NPZ_DIR" \
    --file_name "$FILE_NAME" \
    --num "$NUM" \
    --resolution "$RESOLUTION"