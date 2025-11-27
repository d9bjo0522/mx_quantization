root_dir=/home/tttpd9bjo/mx_quantization
dit_dir=$root_dir/workloads/DiT
mx_dir=$root_dir/microxscaling

export PYTHONPATH=$root_dir:$dit_dir:$mx_dir

model_dir=/work/tttpd9bjo/diffusion/DiT/DiT-XL-2-256x256
pred_mode=MXINT8
sample_dir=$model_dir/samples/$pred_mode

torchrun --master_port 12345 --nnodes=1 --nproc_per_node=1 sample_ddp.py \
    --model DiT-XL/2 \
    --per-proc-batch-size 1 \
    --num-fid-samples 1000 \
    --num-sampling-steps 100 \
    --ckpt $model_dir/pretrained_models/DiT-XL-2-256x256.pt \
    --image-size 256 \
    --num-classes 1000 \
    --sample-dir $sample_dir \
    --current-num-samples 0 \
    --mx-quant \
    --top-k \
    --k 154 \
    --approx-flag \
    --pred-mode "$pred_mode"

## argument explanation
# --sample-dir: directory to save the generated images
# --mx-quant: whether to use MXINT8 quantization
# --top-k: whether to use top-k
# --k: the value of k for top-k
# --approx-flag: whether to use approximate top-k or MXINT8 top-k
# --pred-mode: the mode of prediction (ex_pred (proposed), MXINT4 (Sanger), two_step_leading_ones (EXION), ELSA)