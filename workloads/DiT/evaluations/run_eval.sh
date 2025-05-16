ref_batch_dir="/work/tttpd9bjo/diffusion/DiT/DiT-XL-2-256x256/evaluation"
sample_batch_dir="/work/tttpd9bjo/diffusion/DiT/DiT-XL-2-256x256/samples"

ref_batch_file_name="VIRTUAL_imagenet256_labeled.npz"
sample_batch_file_name="DiT-XL-2-DiT-XL-2-256x256-size-256-vae-ema-cfg-1.5-seed-0-num_samples-15000-w8a8-top128-ex-pred.npz"

ref_batch_file_path="${ref_batch_dir}/${ref_batch_file_name}"
sample_batch_file_path="${sample_batch_dir}/${sample_batch_file_name}"

python evaluator.py "${ref_batch_file_path}" "${sample_batch_file_path}"







