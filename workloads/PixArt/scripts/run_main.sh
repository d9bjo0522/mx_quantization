LOG='../outputs/'
PROMPT_PATH='../prompts/sample.txt'

# Add both paths to PYTHONPATH
# export PYTHONPATH=$PYTHONPATH:/home/tttpd9bjo/mx_quantization/workloads/PixArt:/home/tttpd9bjo/mx_quantization/microxscaling

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

python local_inference.py \
    --log ${LOG} \
    --prompt ${PROMPT_PATH} \
    --batch-size 1 \
    # --mx-quant \
    # --self-top-k \
    # --self-k 3686 \
    # --ex-pred