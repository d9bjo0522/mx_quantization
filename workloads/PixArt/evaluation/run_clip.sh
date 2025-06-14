IMG_PTH='/work/tttpd9bjo/diffusion/PixArt/PixArt-XL-2/evaluation/PixArt-XL-2-256x256/generated_images/w8a8'
PROMPT_PTH='../prompts/coco2017_val5000.txt'



python clip_score.py \
    --img-pth ${IMG_PTH} \
    --prompt-pth ${PROMPT_PTH}