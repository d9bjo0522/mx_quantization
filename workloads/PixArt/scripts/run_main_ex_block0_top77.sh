LOG='/work/tttpd9bjo/diffusion/PixArt/PixArt-XL-2/evaluation/PixArt-XL-2-256x256/generated_images/except_27_block/exclude_block0/ex_top77'
pretrained_models_dir=/work/tttpd9bjo/diffusion/PixArt/PixArt-XL-2/pretrained_models

PROMPT_PATH='../prompts/coco2017_val5000.txt'
# PROMPT_PATH='../prompts/sample_test.txt'

# Add both paths to PYTHONPATH
pixart_dir=/home/tttpd9bjo/mx_quantization/workloads/PixArt

if [[ ":$PYTHONPATH:" != *":$pixart_dir:"* ]]; then
    export PYTHONPATH=$pixart_dir:$PYTHONPATH
fi

# Loop over different start-idx values
for start_idx in 0 1000 2000 3000 4000; do
    echo "Running top77 true top-k except 27th block no top-k and block0 partial_K"
    echo "Running with start-idx = $start_idx"
    python text_local_inference_alpha_ex0_blocks.py \
        --pretrained-models-dir $pretrained_models_dir \
        --image-dir ${LOG} \
        --prompt ${PROMPT_PATH} \
        --resolution 256 \
        --start-idx $start_idx \
        --batch-size 100 \
        --mx-quant \
        --self-top-k \
        --self-k 77 \
        --ex-pred \
        --pred-mode "ex_pred" \
        --exclude-blocks-type "partial_K"
    echo "Completed start-idx = $start_idx"
done