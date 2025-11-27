EVAL_ROOT_DIR="/work/tttpd9bjo/diffusion/DiT/DiT-XL-2-256x256/evaluation"
SAMPLE_ROOT_DIR="/work/tttpd9bjo/diffusion/DiT/DiT-XL-2-256x256/samples"


# REF_FILE_NAME="VIRTUAL_imagenet256_labeled.npz"
REF_FILE_NAME="ImageNet256_10000.npz"
SAMPLE_NAME="true"
TOP_K=141

REF_FILE_PATH="${EVAL_ROOT_DIR}/${REF_FILE_NAME}"
SAMPLE_FILE_PATH="${SAMPLE_ROOT_DIR}/${SAMPLE_NAME}/top${TOP_K}"
# SAMPLE_FILE_PATH="${SAMPLE_ROOT_DIR}/${SAMPLE_NAME}"

SAMPLE_NUM=10000
SAMPLE_NPZ_DIR="${EVAL_ROOT_DIR}/sample_npz"

## compress to npz files first
python to_NPZ.py --sample_dir $SAMPLE_FILE_PATH --npz_dir $SAMPLE_NPZ_DIR --num $SAMPLE_NUM --file_name $SAMPLE_NAME"_top_${TOP_K}"
# python to_NPZ.py --sample_dir $SAMPLE_FILE_PATH --npz_dir $SAMPLE_NPZ_DIR --num $SAMPLE_NUM --file_name $SAMPLE_NAME


## run evaluation for (IS, FID)
python evaluator.py "${REF_FILE_PATH}" "${SAMPLE_NPZ_DIR}/${SAMPLE_NAME}_top_${TOP_K}.npz"
# python evaluator.py "${REF_FILE_PATH}" "${SAMPLE_NPZ_DIR}/${SAMPLE_NAME}.npz"