LOG='../outputs/'
PROMPT_PATH='../prompts/sample.txt'

# Add both paths to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/home/tttpd9bjo/mx_quantization/workloads/PixArt:/home/tttpd9bjo/mx_quantization/microxscaling

# echo "PYTHONPATH: $PYTHONPATH"

# python fp_inference.py \
#     --log ${LOG} \
#     --prompt ${PROMPT_PATH} \
#     --batch-size 1

# # Run the script using python -m for better module resolution
# python mx_inference.py \
#     --log ${LOG} \
#     --prompt ${PROMPT_PATH} \
#     --batch-size 1 \
#     --mx-quant \
#     # --self-top-k \
#     # --self-k 1024 \
#     # --ex-pred
#     # --cross-top-k \
#     # --cross-k 21 \
#     # --ex-pred

# python local_inference.py \
#     --log ${LOG} \
#     --prompt ${PROMPT_PATH} \
#     --batch-size 1 \
#     --mx-quant \
#     --self-top-k \
#     --self-k 4096 \
#     # --ex-pred

python text_local_inference_sigma.py \
    --log ${LOG} \
    --prompt ${PROMPT_PATH} \
    --resolution 1024 \
    --start-idx 1 \
    --batch-size 1 \
    --self-top-k \
    --self-k 2867 \
    --ex-pred

# python text_local_inference_alpha.py \
#     --log ${LOG} \
#     --prompt ${PROMPT_PATH} \
#     --start-idx 1 \
#     --batch-size 1 \
#     --mx-quant \
#     --self-top-k \
#     --self-k 410 \
#     --ex-pred