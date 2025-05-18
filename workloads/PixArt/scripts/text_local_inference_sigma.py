import os
os.environ["HF_HUB_OFFLINE"] = "1"

import json, torch
import random
import numpy as np
import diffusers
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler
)
from safetensors.torch import load_file
from transformers import T5EncoderModel, T5TokenizerFast
from diffusers import PixArtSigmaPipeline
import argparse
import gc


from models.pixart_transformer_2d import MXPixArtTransformer2DModel

def print_gpu_memory(label=""):
    if torch.cuda.is_available():
        print(f"{label} GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"{label} GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_cfg(path):
    with open(path) as f:
        return json.load(f)

## Main function

## instantiate the models
## transformer
def main(args):
    seed_everything(args.seed)

    folder = "/work/tttpd9bjo/diffusion/PixArt/PixArt-Sigma-XL-2"
    transformer_folder = f"{folder}/PixArt-Sigma-XL-2-1024-MS" if args.resolution == 1024 else f"{folder}/PixArt-Sigma-XL-2-2K-MS"

    image_path = "sigma-1024/mx_quant/sf_later"
    ## sample images from prompts
    prompt_path = args.prompt if args.prompt is not None else "./prompts.txt"
    prompts = []
    with open(prompt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            prompts.append(line.strip())
    N_batch = len(prompts) // args.batch_size

    print(f"Found {len(prompts)} prompts, will process in {N_batch} batches")

    ## text encoder
    ## generated text first
    text_encoder = T5EncoderModel.from_pretrained(
        f"{folder}/text_encoder",
        local_files_only=True,
        load_in_8bit=False,     ## set true saves 2GB but slows down 22s
        device_map="auto")
    
    pipe = PixArtSigmaPipeline.from_pretrained(
        f"{folder}", 
        text_encoder=text_encoder, 
        transformer=None, 
        device_map="balanced")

    all_prompts_embeds = []
    all_prompt_attention_masks = []
    all_negative_prompt_embeds = []
    all_negative_prompt_attention_masks = []

    for i in range(N_batch):
        with torch.no_grad():
            prompts_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = pipe.encode_prompt(prompts[i*args.batch_size: (i+1)*args.batch_size])
        all_prompts_embeds.append(prompts_embeds)
        all_prompt_attention_masks.append(prompt_attention_mask)
        all_negative_prompt_embeds.append(negative_prompt_embeds)
        all_negative_prompt_attention_masks.append(negative_prompt_attention_mask)
    
    del pipe, text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    tr_cfg_dict = read_cfg(f"{transformer_folder}/transformer/config.json")
    transformer = MXPixArtTransformer2DModel.from_config(tr_cfg_dict)
    print(f"Initial model configs: mx_quant={transformer.mx_quant}, mx_specs={transformer.mx_specs}, self_top_k={transformer.self_top_k}, self_k={transformer.self_k}, ex_pred={transformer.ex_pred}")

    ## set the model configs
    mx_specs = {
        'w_elem_format': 'int8',
        'a_elem_format': 'int8',
        'scale_bits': 8,
        'shared_exp_method': 'max',
        'block_size': 32,
        'bfloat': 32,
        'fp': 0,
        'bfloat_subnorms': True,
        'round': 'nearest',
        'round_mx_output': 'nearest',
        'round_output': 'nearest',
        'round_weight': 'nearest',
        'mx_flush_fp32_subnorms': True,
        'custom_cuda': False,
        'quantize_backprop': False,
    }
    # Apply MX quantization settings to reduce memory usage
    transformer.set_config(
        mx_quant=args.mx_quant, 
        mx_specs=mx_specs, 
        self_top_k=args.self_top_k, 
        self_k=args.self_k, 
        ex_pred=args.ex_pred
    )
    print(f"Model configs: mx_quant={transformer.transformer_blocks[0].mx_quant}, mx_specs={transformer.transformer_blocks[0].mx_specs}, self_top_k={transformer.transformer_blocks[0].self_top_k}, self_k={transformer.transformer_blocks[0].self_k}, ex_pred={transformer.transformer_blocks[0].ex_pred}")

    transformer_checkpoint = load_file(
        f"{transformer_folder}/transformer/diffusion_pytorch_model.safetensors",
        device="cpu"
    )
    transformer.load_state_dict(transformer_checkpoint)

    # Load other components
    print("Loading other components...")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        f"{folder}/scheduler", 
        local_files_only=True
    )
    
    vae = AutoencoderKL.from_pretrained(
        f"{folder}/vae", 
        local_files_only=True
    )
    
    tokenizer = T5TokenizerFast.from_pretrained(
        f"{folder}/tokenizer", 
        local_files_only=True
    )
    
    # Create minimal pipeline
    print("Building pipeline...")
    dtype = torch.float32

    pipe = PixArtSigmaPipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=None,
        tokenizer=tokenizer,
        scheduler=scheduler,
    ).to(dtype=dtype)

    ## reduce memory usage
    pipe.enable_model_cpu_offload()
    # pipe.enable_attention_slicing("max")
    pipe.enable_attention_slicing()
    # pipe.enable_sequential_cpu_offload(gpu_id=0)        # streams weights & acts
    pipe.vae.enable_tiling()

    # print_gpu_memory("After loading pipe")
    for i in range(N_batch):
        print(f"\nProcessing batch {i+1}/{N_batch}")
        
        # Get pre-computed embeddings for this batch
        prompt_embeds = all_prompts_embeds[i]
        prompt_attention_mask = all_prompt_attention_masks[i]
        negative_prompt_embeds = all_negative_prompt_embeds[i]
        negative_prompt_attention_mask = all_negative_prompt_attention_masks[i]
                
        # print_gpu_memory(f"Before generating batch {i+1}")
        images = pipe(
            negative_prompt = None,
            prompt_embeds = prompt_embeds,
            prompt_attention_mask = prompt_attention_mask,
            negative_prompt_embeds = negative_prompt_embeds,
            negative_prompt_attention_mask = negative_prompt_attention_mask,
            num_images_per_prompt = 1,
            num_inference_steps=args.num_sampling_steps,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).images

        print(f"Export image of batch {i}")
        save_path = os.path.join(args.log, image_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i_image in range(args.batch_size):
            images[i_image].save(os.path.join(save_path, f"{args.start_idx + i_image + args.batch_size*i}.jpg"))
        
        
        ## added to avoid OOM issue
        # Delete the images variable to free up memory (already saved)
        del images, prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask
        gc.collect()
        torch.cuda.empty_cache()
        # print_gpu_memory(f"After generating batch {i+1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=20)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default=None)

    parser.add_argument("--resolution", type=int, default=1024)
    ## Added for MX quantization
    parser.add_argument("--mx-quant", action="store_true", default=False)
    parser.add_argument("--self-top-k", action="store_true", default=False)
    parser.add_argument("--self-k", type=int, default=20)
    parser.add_argument("--ex-pred", action="store_true", default=False)
    parser.add_argument("--start-idx", type=int, default=0)
    args = parser.parse_args()
    main(args)



