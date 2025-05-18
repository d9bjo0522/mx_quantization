import torch
import os
import sys
import diffusers
import time
import shutil
import argparse
import logging
import random
import numpy as np

from diffusers import PixArtSigmaPipeline
# from qdiff.utils import apply_func_to_submodules, seed_everything, setup_logging
# from models.pipeline_pixart_sigma import PixArtSigmaPipeline

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from collections import defaultdict

def list_fp32_params(pipe):
    """
    Return {component_name: [param_full_name, …]} for params that are fp32
    """
    fp32 = defaultdict(list)

    # diffusers ≥ 0.25 has `pipe.components`; fall back to dir() otherwise
    comp_dict = getattr(pipe, "components", None)
    if comp_dict is None:                        # older versions
        comp_dict = {n: getattr(pipe, n) for n in dir(pipe)}

    for comp_name, obj in comp_dict.items():
        if isinstance(obj, torch.nn.Module):
            for p_name, p in obj.named_parameters(recurse=True):
                if p.dtype == torch.float32:
                    fp32[f"{comp_name}"].append(f"{comp_name}.{p_name}")
    return fp32

def main(args):
    seed_everything(args.seed)
    torch.set_grad_enabled(False)
    device="cuda" if torch.cuda.is_available() else "cpu"

    # ckpt_path = args.ckpt if args.ckpt is not None else "./pretrained_models/"
    pipe = PixArtSigmaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        # ckpt_path,
        torch_dtype=torch.float16  # due to CUDA kernel only supports fp16, we donot use bfloat16 here. 
    ).to(device)

    fp32_layers = list_fp32_params(pipe)
    for comp, names in fp32_layers.items():
        print(f"{comp:<15}  —  {len(names)} fp32 tensors")
        for n in names:
            print("   ", n)

    # INFO: if memory intense
    # pipe.enable_model_cpu_offload()
    # pipe.vae.enable_tiling()
    
    # read the promts
    prompt_path = args.prompt if args.prompt is not None else "./prompts.txt"
    prompts = []
    with open(prompt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            prompts.append(line.strip())

    N_batch = len(prompts) // args.batch_size # drop_last
    for i in range(N_batch):
        images = pipe(
            prompt=prompts[i*args.batch_size: (i+1)*args.batch_size],
            num_inference_steps=args.num_sampling_steps,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).images
        print(f"Export image of batch {i}")
        save_path = os.path.join(args.log, "generated_images")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i_image in range(args.batch_size):
            images[i_image].save(os.path.join(save_path, f"{i_image + args.batch_size*i}.jpg"))
        
        
        ## added to avoid OOM issue
        # Delete the images variable to free up memory
        del images

        # Clear the GPU cache
        torch.cuda.empty_cache()
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str)
    parser.add_argument("--cfg-scale", type=float, default=4.5)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    main(args)