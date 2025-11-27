#export CUDA_VISIBLE_DEVICES=0
root_dir=/home/tttpd9bjo/mx_quantization
dit_dir=$root_dir/workloads/DiT
mx_dir=$root_dir/microxscaling
script_dir=$dit_dir/scripts
export PYTHONPATH=$root_dir:$dit_dir:$mx_dir:$script_dir

mode=MXINT4
model_dir=/work/tttpd9bjo/diffusion/DiT/DiT-XL-2-256x256
sample_dir=$model_dir/samples/thesis_use

mkdir -p $sample_dir

## DiT-XL/2 256x256

python sample.py \
    --model DiT-XL/2 \
    --ckpt $model_dir/pretrained_models/DiT-XL-2-256x256.pt \
    --image-size 256 \
    --vae mse \
    --num-sampling-steps 100 \
    --seed 0 \
    --sample-dir $sample_dir \
    --mx-quant \
    --top-k \
    --k 154 \
    --approx-flag \
    --pred-mode "$mode"

## argument explanation
# --sample-dir: directory to save the generated images
# --mx-quant: whether to use MXINT8 quantization
# --top-k: whether to use top-k
# --k: the value of k for top-k
# --approx-flag: whether to use approximate top-k or MXINT8 top-k
# --pred-mode: the mode of prediction (ex_pred (proposed), MXINT4 (Sanger), two_step_leading_ones (EXION), ELSA)


## DiT-XL/2 512x512

# python sample.py \
#     --model DiT-XL/2 \
#     --ckpt /work/tttpd9bjo/diffusion/DiT/DiT-XL-2-512x512/pretrained_models/DiT-XL-2-512x512.pt \
#     --image-size 512 \
#     --vae mse \
#     --num-sampling-steps 250 \
#     --seed 0 \
#     --sample-dir sample_512_mx16_topk128_ex \
#     --mx-quant True \
#     --top-k True \
#     --k 128 \
#     --ex-pred True
    