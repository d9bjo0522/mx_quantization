# torchrun --nnodes=1 --nproc_per_node=8 sample_ddp.py \
#  --model DiT-XL/2 \
#  --num-fid-samples 50000 \
#  --ckpt /work/tttpd9bjo/diffusion/DiT/DiT-XL-2-256x256/pretrained_models/DiT-XL-2-256x256.pt \
#  --image-size 256 \
#  --num-classes 1000 \
#  --sample-dir /work/tttpd9bjo/diffusion/DiT/DiT-XL-2-256x256/samples

#export CUDA_VISIBLE_DEVICES=0

python sample.py \
    --model DiT-XL/2 \
    --ckpt /work/tttpd9bjo/diffusion/DiT/DiT-XL-2-256x256/pretrained_models/DiT-XL-2-256x256.pt \
    --image-size 256 \
    --vae mse \
    --num-sampling-steps 250 \
    --seed 0
    
    