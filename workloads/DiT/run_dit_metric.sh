root_dir=/home/tttpd9bjo/mx_quantization
dit_dir=$root_dir/workloads/DiT
mx_dir=$root_dir/microxscaling

export PYTHONPATH=$root_dir:$dit_dir:$mx_dir

model_dir=/work/tttpd9bjo/diffusion/DiT/DiT-XL-2-256x256
sample_name=w8a8
sample_dir=$model_dir/samples/$sample_name

torchrun --master_port 12345 --nnodes=1 --nproc_per_node=8 sample_ddp.py \
 --model DiT-XL/2 \
 --per-proc-batch-size 50 \
 --num-fid-samples 10000 \
 --num-sampling-steps 50 \
 --ckpt $model_dir/pretrained_models/DiT-XL-2-256x256.pt \
 --image-size 256 \
 --num-classes 1000 \
 --sample-dir $sample_dir \
 --current-num-samples 0 \
 --mx-quant
#  --top-k \
#  --k 154 \
#  --ex-pred \
#  --pred-mode "ex_pred"
