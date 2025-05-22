IMG_PTH='../evaluation/generated_images/alpha-512/top512/true/'
PROMPT_PTH='../prompts/sample.txt'



python ../evaluation/clip_score.py \
    --img-pth ${IMG_PTH} \
    --prompt-pth ${PROMPT_PTH}