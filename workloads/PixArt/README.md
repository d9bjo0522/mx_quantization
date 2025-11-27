# PixArt Image Generation and Evaluation

## PixArt conda environment setup
``` 
cd environments/ 
source run_env.sh
```
```environment.yml```: conda environment for running image generation \
```../evaluation/requirements.txt```: required packages for evaluation \
```run_env.sh```: install packages for both image generation and evaluation

## Overview
- Apply MX quantization and top-k attention pruning on PixArt models
  - ./models/: directory for changed attention modules
  - ../../funcs/: directory for different approximation related (proposed, Sanger, EXION, ELSA) modules
- Run 256x256 image generation
  - ./scripts/: directory for running image generation process
    - ```test_local_inference_alpha.py```, ```run_pixart_alpha.sh```: main function of pixart-alpha 256x256 image generation
- Evaluate generated images
  - ./evaluation/: directory for image generation metric evaluation
  - ```run_all_eval.sh```: run FID
    - Evaluate through comparing reference and generated images
      - Need reference compressed npz file first
      - Specify the generated image directory
        - This script will compress the generated images to an npz file
      - Acquire FID results through comparing reference and generated npz files

## Fast implementation process
```
cd ./scripts
source run_pixart_alpha.sh

cd ../evaluations
source run_all_eval.sh
```
