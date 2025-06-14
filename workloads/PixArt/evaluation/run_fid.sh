ref_batch_dir="/work/tttpd9bjo/diffusion/PixArt/PixArt-XL-2/evaluation/PixArt-XL-2-256x256/"
sample_batch_dir="/work/tttpd9bjo/diffusion/PixArt/PixArt-XL-2/evaluation/PixArt-XL-2-256x256/fid"

ref_batch_file_name="val2017_5000.npz"
sample_batch_file_name="w8a8.npz"



ref_batch_file_path="${ref_batch_dir}/${ref_batch_file_name}"
sample_batch_file_path="${sample_batch_dir}/${sample_batch_file_name}"


python FID_score.py "${ref_batch_file_path}" "${sample_batch_file_path}"







