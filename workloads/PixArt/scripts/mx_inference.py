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

# from diffusers import PixArtSigmaPipeline, Transformer2DModel
# from qdiff.utils import apply_func_to_submodules, seed_everything, setup_logging
from models.pipeline_pixart_sigma import MXPixArtSigmaPipeline
from models.pixart_transformer_2d import MXPixArtTransformer2DModel
from models.customize_transformer_block import MXSelfAttention, MXCrossAttention

from diffusers import PixArtSigmaPipeline, Transformer2DModel
from diffusers.models import PixArtTransformer2DModel
## monkey patch
diffusers.models.PixArtTransformer2DModel = MXPixArtTransformer2DModel
diffusers.PixArtSigmaPipeline = MXPixArtSigmaPipeline
# diffusers.Transformer2DModel = MXPixArtTransformer2DModel

from diffusers import PixArtSigmaPipeline   ## need to reload to make sure the patch is applied


# diffusers.Transformer2DModel = MXPixArtTransformer2DModel
# pixart_transformer_2d.PixArtTransformer2DModel = MXPixArtTransformer2DModel

# from diffusers import PixArtSigmaPipeline, Transformer2DModel
# from diffusers.models import PixArtTransformer2DModel
# print(PixArtSigmaPipeline._component_names)
# print(PixArtSigmaPipeline._optional_components)
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# def apply_mx_settings(pipe, *, mx_quant, mx_specs, self_top_k, self_k, cross_top_k, cross_k, ex_pred):
#     """
#     Post-hoc switch that pushes the MX flags down to every layer.
#     Call it immediately after from_pretrained().
    
#     Parameters:
#         pipe: The pipeline object
#         mx_quant: Whether to use MX quantization
#         mx_specs: MX specifications dictionary
#         self_top_k: Whether to use top-k for self-attention
#         self_k: k value for self-attention
#         cross_top_k: Whether to use top-k for cross-attention
#         cross_k: k value for cross-attention
#         ex_pred: Whether to use exponent-based prediction
#     """
#     def _propagate(module):
#         for m in module.modules():
#             if hasattr(m, "mx_quant"):
#                 m.mx_quant = mx_quant
#             if hasattr(m, "mx_specs"):
#                 m.mx_specs = mx_specs
            
#             # Apply different settings for self-attention and cross-attention
#             if hasattr(m, "top_k"):
#                 if isinstance(m, MXSelfAttention):
#                     m.top_k = self_top_k
#                     # print(f"Self attention top-k: {m.top_k}")
#                 elif isinstance(m, MXCrossAttention):
#                     m.top_k = cross_top_k
                    
#             if hasattr(m, "k"):
#                 if isinstance(m, MXSelfAttention):
#                     m.k = self_k
#                 elif isinstance(m, MXCrossAttention):
#                     m.k = cross_k
                    
#             if hasattr(m, "ex_pred"):
#                 m.ex_pred = ex_pred

#     _propagate(pipe.transformer)
#     # pipe.transformer.set_config(mx_quant=mx_quant, mx_specs=mx_specs)
#     # ── optional: rebuild the blocks if we actually toggled the mode ──
#     # root = pipe.transformer
#     # if hasattr(root, "_replace_transformer_blocks"):
#     #     current = root.transformer_blocks[0].mx_quant   # int or bool
#     #     if current != mx_quant:
#     #         root._replace_transformer_blocks()          # copies weights back in

def main(args):
    seed_everything(args.seed)
    torch.set_grad_enabled(False)
    device="cuda" if torch.cuda.is_available() else "cpu"

    # if args.log is not None:
    #     if not os.path.exists(args.log):
    #         os.makedirs(args.log)
    # log_file = os.path.join(args.log, 'run.log')
    # setup_logging(log_file)
    # logger = logging.getLogger(__name__)

    # ckpt_path = args.ckpt if args.ckpt is not None else "./pretrained_models/"
    mx_specs = {
        'w_elem_format': 'int4',
        'a_elem_format': 'int4',
        'scale_bits': 8,
        'shared_exp_method': 'max',
        'block_size': 32,
        'bfloat': 16,
        'fp': 0,
        'bfloat_subnorms': True,
        'round': 'nearest',
        'round_mx_output': 'nearest',
        'round_output': 'nearest',
        'round_weight': 'nearest',
        'mx_flush_fp32_subnorms': False,
        'custom_cuda': False,
        'quantize_backprop': False,
    }

    # transformer = Transformer2DModel.from_pretrained(
    #     "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", 
    #     subfolder='transformer', 
    #     torch_dtype=torch.float16,
    #     use_safetensors=True,
    # )
    # transformer = PixArtTransformer2DModel.from_pretrained(
    #     "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    #     subfolder="transformer",
    #     torch_dtype=torch.float16,
    #     mx_quant=True,
    #     mx_specs=mx_specs,
    #     self_top_k=args.self_top_k,
    #     self_k=args.self_k,
    #     cross_top_k=args.cross_top_k,
    #     cross_k=args.cross_k,
    #     ex_pred=args.ex_pred,
    # )
    # pipe = PixArtSigmaPipeline.from_pretrained(
    #     "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
    #     transformer=transformer,
    #     # ckpt_path,
    #     torch_dtype=torch.float16,  # due to CUDA kernel only supports fp16, we donot use bfloat16 here. 
    #     # use_safetensors=True,
    # ).to(device)
    pipe = PixArtSigmaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        # ckpt_path,
        torch_dtype=torch.float16,  # due to CUDA kernel only supports fp16, we donot use bfloat16 here. 
        # use_safetensors=True,
    ).to(device)

    # apply_mx_settings(
    #     pipe,
    #     mx_quant = args.mx_quant,
    #     mx_specs = mx_specs,
    #     self_top_k = args.self_top_k,
    #     self_k = args.self_k,
    #     cross_top_k = args.cross_top_k,
    #     cross_k = args.cross_k,
    #     ex_pred = args.ex_pred,
    # )

    # INFO: if memory intense, use this to offload the model to CPU
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    ## Added for MX quantization
    model = pipe.transformer

    ## Model configs
    print(f"Model type: {type(model).__name__}")
    print(f"Initial model configs: mx_quant={model.mx_quant}, mx_specs={model.mx_specs}, self_top_k={model.self_top_k}, self_k={model.self_k}, cross_top_k={model.cross_top_k}, cross_k={model.cross_k}, ex_pred={model.ex_pred}")

    ## set the MX quantization configs
    print(f"Setting the MX quantization configs")
    model.set_config(mx_quant=args.mx_quant, mx_specs=mx_specs, self_top_k=args.self_top_k, self_k=args.self_k, cross_top_k=args.cross_top_k, cross_k=args.cross_k, ex_pred=args.ex_pred)
    print(f"Model configs: mx_quant={model.mx_quant}, mx_specs={model.mx_specs}, self_top_k={model.self_top_k}, self_k={model.self_k}, cross_top_k={model.cross_top_k}, cross_k={model.cross_k}, ex_pred={model.ex_pred}")
    
    # Store original number of parameters before modification
    original_param_count = sum(p.numel() for p in model.parameters())
    print(f"Original model parameter count: {original_param_count:,}")
    
    # Apply MX quantization configurations
    # model.mx_quant = args.mx_quant
    # model.mx_specs = mx_specs
    # model.top_k = args.top_k
    # model.k = args.k
    # model.ex_pred = args.ex_pred
        
    print(f"Configuration: MX Quant={args.mx_quant}, Self Top-K={args.self_top_k}, Self K={args.self_k}, Cross Top-K={args.cross_top_k}, Cross K={args.cross_k}, Ex-Pred={args.ex_pred}")

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
        save_path = os.path.join(args.log, "generated_images/mx_quant")
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