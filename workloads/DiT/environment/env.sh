# create conda environment (to run DiT image generation)
conda env create -f environment.yml

# activate environment
conda activate DiT

# install requirements for evaluation
pip install -r ../evaluations/requirements.txt