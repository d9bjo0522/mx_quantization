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
# from diffusers.utils import get_memory_optimized_attention_text


# from models.pipeline_pixart_sigma import MXPixArtSigmaPipeline
from models.pixart_transformer_2d import MXPixArtTransformer2DModel, PixArtTransformer2DModel

print("Transformer now points to:", PixArtTransformer2DModel)
print("Pipeline    now points to:", PixArtSigmaPipeline)

folder = "/work/tttpd9bjo/diffusion/PixArt/PixArt-Sigma-XL-2-1024-MS"

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


## instantiate the models
## transformer
def main(args):
    seed_everything(args.seed)
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
    transformer.set_config(mx_quant=args.mx_quant, mx_specs=mx_specs, self_top_k=args.self_top_k, self_k=args.self_k, cross_top_k=args.cross_top_k, cross_k=args.cross_k, ex_pred=args.ex_pred)
    print(f"Model configs: mx_quant={transformer.transformer_blocks[0].mx_quant}, mx_specs={transformer.transformer_blocks[0].mx_specs}, self_top_k={transformer.transformer_blocks[0].self_top_k}, self_k={transformer.transformer_blocks[0].self_k}, cross_top_k={transformer.transformer_blocks[0].cross_top_k}, cross_k={transformer.transformer_blocks[0].cross_k}, ex_pred={transformer.transformer_blocks[0].ex_pred}")

    transformer_checkpoint = load_file(
        f"{folder}/transformer/diffusion_pytorch_model.safetensors",
        device="cpu"
    )

    transformer.load_state_dict(transformer_checkpoint)
    dtype   = torch.float16
    device  = torch.device("cuda")

    ## text-encoder
    ## to(torch.float16): makes the parameters float16
    # transformer = PixArtTransformer2DModel.from_pretrained(f"{folder}/transformer", local_files_only=True)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(f"{folder}/scheduler", local_files_only=True)
    vae = AutoencoderKL.from_pretrained(f"{folder}/vae", local_files_only=True)
    text_encoder = T5EncoderModel.from_pretrained(f"{folder}/text_encoder", local_files_only=True)
    tokenizer = T5TokenizerFast.from_pretrained(f"{folder}/tokenizer", local_files_only=True)


    ## test the model
    pipe = PixArtSigmaPipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,        # let diffusers infer from folder path
        scheduler=scheduler,
    ).to(dtype = dtype)

    pipe.text_encoder.to(dtype=torch.float32)

    orig_encode = pipe.encode_prompt
    def _encode_prompt_fp16(*args, **kwargs):
        prompts_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = orig_encode(*args, **kwargs)
        return prompts_embeds.to(dtype=torch.float16), prompt_attention_mask.to(dtype=torch.float16), negative_prompt_embeds.to(dtype=torch.float16), negative_prompt_attention_mask.to(dtype=torch.float16)
    pipe.encode_prompt = _encode_prompt_fp16

    ## reduce memory usage
    pipe.enable_model_cpu_offload()
    # pipe.enable_attention_slicing("max")
    pipe.enable_attention_slicing()
    # pipe.enable_sequential_cpu_offload(gpu_id=0)        # streams weights & acts
    pipe.vae.enable_tiling()

    ## sample images from prompts
    prompt_path = args.prompt if args.prompt is not None else "./prompts.txt"
    prompts = []
    with open(prompt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            prompts.append(line.strip())

    N_batch = len(prompts) // args.batch_size

    for i in range(N_batch):
        images = pipe(
            prompt=prompts[i*args.batch_size: (i+1)*args.batch_size],
            num_inference_steps=args.num_sampling_steps,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).images
        print(f"Export image of batch {i}")
        save_path = os.path.join(args.log, "generated_images/mx_quant")
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i_image in range(args.batch_size):
            images[i_image].save(os.path.join(save_path, f"{i_image + args.batch_size*i}.jpg"))
        
        
        ## added to avoid OOM issue
        # Delete the images variable to free up memory (already saved)
        del images
        # Clear the GPU cache
        torch.cuda.empty_cache()

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
    args = parser.parse_args()
    main(args)



