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
from diffusers import Transformer2DModel
import argparse
import gc


from models.pixart_transformer_2d import MXPixArtTransformer2DModel, PixArtTransformer2DModel

# print("Transformer now points to:", PixArtTransformer2DModel)
# print("Pipeline    now points to:", PixArtSigmaPipeline)

folder = "/work/tttpd9bjo/diffusion/PixArt/PixArt-Sigma-XL-2-1024-MS"

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
    
# # Separate text encoder function that runs fully on CPU
# def encode_prompts_on_cpu(prompts, batch_size=1, folder_path=None):
#     print("Starting text encoding on CPU...")
    
#     # Force CPU for text encoder
#     device = torch.device("cpu")
    
#     # Load models to CPU explicitly
#     tokenizer = T5TokenizerFast.from_pretrained(f"{folder_path}/tokenizer", local_files_only=True)
#     text_encoder = T5EncoderModel.from_pretrained(
#         f"{folder_path}/text_encoder", 
#         local_files_only=True,
#         device_map={"": device}  # Explicitly map to CPU
#     )
    
#     print_gpu_memory("After loading text encoder (should be minimal)")
    
#     # Process in batches
#     n_batch = len(prompts) // batch_size
#     all_results = []
    
#     for i in range(n_batch):
#         batch_prompts = prompts[i*batch_size: (i+1)*batch_size]
        
#         # Tokenize text
#         text_inputs = tokenizer(
#             batch_prompts,
#             padding="max_length",
#             max_length=tokenizer.model_max_length,
#             truncation=True,
#             return_tensors="pt"
#         ).to(device)
        
#         uncond_input = tokenizer(
#             [""] * len(batch_prompts),
#             padding="max_length",
#             max_length=tokenizer.model_max_length,
#             truncation=True,
#             return_tensors="pt"
#         ).to(device)
        
#         # Extract embeddings
#         with torch.no_grad():
#             prompt_embeds = text_encoder(
#                 text_inputs.input_ids,
#                 attention_mask=text_inputs.attention_mask
#             )[0]
            
#             negative_prompt_embeds = text_encoder(
#                 uncond_input.input_ids,
#                 attention_mask=uncond_input.attention_mask
#             )[0]
            
#         # Convert to float16 for storage efficiency
#         prompt_embeds = prompt_embeds.to(torch.float16)
#         negative_prompt_embeds = negative_prompt_embeds.to(torch.float16)
        
#         # Save tuple of (prompt_embeds, attention_mask, negative_prompt_embeds, negative_attention_mask)
#         all_results.append((
#             prompt_embeds, 
#             text_inputs.attention_mask.to(torch.float16), 
#             negative_prompt_embeds,
#             uncond_input.attention_mask.to(torch.float16)
#         ))
        
#     # Clean up
#     del text_encoder, tokenizer
#     gc.collect()
#     torch.cuda.empty_cache()
    
#     print("Text encoding completed on CPU")
#     return all_results

## Main function

## instantiate the models
## transformer
def main(args):
    seed_everything(args.seed)

    ## sample images from prompts
    prompt_path = args.prompt if args.prompt is not None else "./prompts.txt"
    prompts = []
    with open(prompt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            prompts.append(line.strip())
    N_batch = len(prompts) // args.batch_size

    print(f"Found {len(prompts)} prompts, will process in {N_batch} batches")
    # print_gpu_memory("Before any model loading")

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

    tr_cfg_dict = read_cfg(f"{folder}/transformer/config.json")
    transformer = MXPixArtTransformer2DModel.from_config(tr_cfg_dict)
    print(f"Initial model configs: mx_quant={transformer.mx_quant}, mx_specs={transformer.mx_specs}, self_top_k={transformer.self_top_k}, self_k={transformer.self_k}, cross_top_k={transformer.cross_top_k}, cross_k={transformer.cross_k}, ex_pred={transformer.ex_pred}")

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
    # transformer.set_config(mx_quant=args.mx_quant, mx_specs=mx_specs, self_top_k=args.self_top_k, self_k=args.self_k, cross_top_k=args.cross_top_k, cross_k=args.cross_k, ex_pred=args.ex_pred)
    # Apply MX quantization settings to reduce memory usage
    transformer.set_config(
        mx_quant=args.mx_quant, 
        mx_specs=mx_specs, 
        self_top_k=args.self_top_k, 
        self_k=args.self_k, 
        cross_top_k=args.cross_top_k, 
        cross_k=args.cross_k, 
        ex_pred=args.ex_pred
    )
    print(f"Model configs: mx_quant={transformer.transformer_blocks[0].mx_quant}, mx_specs={transformer.transformer_blocks[0].mx_specs}, self_top_k={transformer.transformer_blocks[0].self_top_k}, self_k={transformer.transformer_blocks[0].self_k}, cross_top_k={transformer.transformer_blocks[0].cross_top_k}, cross_k={transformer.transformer_blocks[0].cross_k}, ex_pred={transformer.transformer_blocks[0].ex_pred}")

    transformer_checkpoint = load_file(
        f"{folder}/transformer/diffusion_pytorch_model.safetensors",
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
    dtype = torch.float16

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
        save_path = os.path.join(args.log, "generated_images/coco_1024/fp16")
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
    ## Added for MX quantization
    parser.add_argument("--mx-quant", action="store_true", default=False)
    parser.add_argument("--self-top-k", action="store_true", default=False)
    parser.add_argument("--self-k", type=int, default=20)
    parser.add_argument("--cross-top-k", action="store_true", default=False)
    parser.add_argument("--cross-k", type=int, default=20)
    parser.add_argument("--ex-pred", action="store_true", default=False)
    parser.add_argument("--start-idx", type=int, default=0)
    args = parser.parse_args()
    main(args)



