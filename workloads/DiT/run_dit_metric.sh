export PYTHONPATH=/home/tttpd9bjo/mx_quantization/microxscaling:$PYTHONPATH

torchrun --nnodes=1 --nproc_per_node=8 sample_ddp.py \
 --model DiT-XL/2 \
 --per-proc-batch-size 10 \
 --num-fid-samples 4000 \
 --num-sampling-steps 50 \
 --ckpt /work/tttpd9bjo/diffusion/DiT/DiT-XL-2-256x256/pretrained_models/DiT-XL-2-256x256.pt \
 --image-size 256 \
 --num-classes 1000 \
 --sample-dir /work/tttpd9bjo/diffusion/DiT/DiT-XL-2-256x256/samples \
 --current-num-samples 12000 \
 --mx-quant \
 --top-k \
 --k 128 \
 --ex-pred
