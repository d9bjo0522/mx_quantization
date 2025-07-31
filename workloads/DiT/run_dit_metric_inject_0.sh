root_dir=/home/tttpd9bjo/mx_quantization
dit_dir=$root_dir/workloads/DiT
mx_dir=$root_dir/microxscaling

export PYTHONPATH=$root_dir:$dit_dir:$mx_dir

model_dir=/work/tttpd9bjo/diffusion/DiT/DiT-XL-2-256x256
base_sample_name=true_top90
block_id=0
sample_dir=$model_dir/samples/inject_noise/block_$block_id/$base_sample_name

    
torchrun --master_port 12345 --nnodes=1 --nproc_per_node=4 sample_ddp_0.py \
    --model DiT-XL/2 \
    --per-proc-batch-size 50 \
    --num-fid-samples 5000 \
    --num-sampling-steps 50 \
    --ckpt $model_dir/pretrained_models/DiT-XL-2-256x256.pt \
    --image-size 256 \
    --num-classes 1000 \
    --sample-dir $sample_dir \
    --current-num-samples 0 \
    --mx-quant \
    --top-k \
    --k 90
