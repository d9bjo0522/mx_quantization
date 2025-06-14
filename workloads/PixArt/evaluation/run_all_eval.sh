## Exit on any error
set -e

RES="256"
TOPK="top140"
SAMPLE_IMG_NAME="27_blocks_true_20ex"
NUM="5000"

if [ -n "$TOPK" ]; then
    SAMPLE_IMG_DIR="${TOPK}/exclude_blocks/${SAMPLE_IMG_NAME}"
    FILE_NAME="${TOPK}_${SAMPLE_IMG_NAME}"
    echo "Running $RES x $RES $TOPK $SAMPLE_IMG_NAME"
else
    SAMPLE_IMG_DIR="${SAMPLE_IMG_NAME}"
    FILE_NAME="${SAMPLE_IMG_NAME}"
    echo "Running $RES x $RES $SAMPLE_IMG_NAME"
fi

echo -e "Running sample number: $NUM\n"

PROMPT_PTH='../prompts/coco2017_val5000.txt'

SAMPLE_DIR="/work/tttpd9bjo/diffusion/PixArt/PixArt-XL-2/evaluation/PixArt-XL-2-${RES}x${RES}/generated_images"
SAMPLE_NPZ_DIR="/work/tttpd9bjo/diffusion/PixArt/PixArt-XL-2/evaluation/PixArt-XL-2-${RES}x${RES}/fid"

REF_NPZ_DIR="/work/tttpd9bjo/diffusion/PixArt/PixArt-XL-2/evaluation/PixArt-XL-2-${RES}x${RES}/val2017_${RES}_5000.npz"
# REF_NPZ_DIR="/work/tttpd9bjo/diffusion/PixArt/PixArt-XL-2/evaluation/PixArt-XL-2-256x256/fid/fp32.npz"

echo "Compress to npz file"
python toNPZ.py \
    --sample_dir "${SAMPLE_DIR}/${SAMPLE_IMG_DIR}" \
    --npz_dir $SAMPLE_NPZ_DIR \
    --file_name $FILE_NAME \
    --num $NUM



echo -e "\nRunning FID score"
echo -e "Reference npz file: $REF_NPZ_DIR"
echo -e "Sample npz file: ${SAMPLE_NPZ_DIR}/${FILE_NAME}.npz"

python FID_score.py $REF_NPZ_DIR "${SAMPLE_NPZ_DIR}/${FILE_NAME}.npz"

# if [ "$NUM" = "5000" ]; then
#     echo "Run clip score"
#     python clip_score.py \
#         --img-pth "${SAMPLE_DIR}/${SAMPLE_IMG_DIR}" \
#         --prompt-pth $PROMPT_PTH
# fi