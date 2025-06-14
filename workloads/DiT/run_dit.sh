# torchrun --nnodes=1 --nproc_per_node=8 sample_ddp.py \
#  --model DiT-XL/2 \
#  --num-fid-samples 50000 \
#  --ckpt /work/tttpd9bjo/diffusion/DiT/DiT-XL-2-256x256/pretrained_models/DiT-XL-2-256x256.pt \
#  --image-size 256 \
#  --num-classes 1000 \
#  --sample-dir /work/tttpd9bjo/diffusion/DiT/DiT-XL-2-256x256/samples

#export CUDA_VISIBLE_DEVICES=0
root_dir=/home/tttpd9bjo/mx_quantization
dit_dir=$root_dir/workloads/DiT
mx_dir=$root_dir/microxscaling
export PYTHONPATH=$root_dir:$dit_dir:$mx_dir

model_dir=/work/tttpd9bjo/diffusion/DiT/DiT-XL-2-256x256
sample_dir=$model_dir/samples

## DiT-XL/2 256x256

python sample.py \
    --model DiT-XL/2 \
    --ckpt $model_dir/pretrained_models/DiT-XL-2-256x256.pt \
    --image-size 256 \
    --vae mse \
    --num-sampling-steps 50 \
    --seed 0 \
    --sample-dir sample_256_mx32_w8a8_top128_mid45t \
    --mx-quant \
    --top-k \
    --k 128 \
    --ex-pred

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
    