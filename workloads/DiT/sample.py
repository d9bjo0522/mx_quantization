# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
    # mx-quantization config
    mx_specs = {
        'w_elem_format': 'int8',
        'a_elem_format': 'int8',
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
    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        mx_quant = args.mx_quant,
        mx_specs = mx_specs,
        top_k = args.top_k,
        k = args.k,
        ex_pred = args.ex_pred
    ).to(device)

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    # class_labels = [207]
   
    # # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
    # Sample images:
    # samples = diffusion.p_sample_loop(
        # model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    # )

    # Progressive sampling with timestep control
    final_sample = None
    for sample_output in diffusion.p_sample_loop_progressive(
        model.forward_with_cfg,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device
    ):
        current_timestep = sample_output["timestep"]
        current_sample = sample_output["sample"]
            
        final_sample = current_sample
        del current_sample
        torch.cuda.empty_cache()
    # Process the final samples
    samples = final_sample
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample
    # Save and display images:
    save_image(samples, f"{args.sample_dir}.png", nrow=4, normalize=True, value_range=(-1, 1))
    del samples, final_sample
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)  # Set to 50 timesteps
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    # mx-quantization configs
    parser.add_argument("--mx-quant", action='store_true')
    parser.add_argument("--sample-dir", type=str, default=None)
    parser.add_argument("--top-k", action='store_true')
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--ex-pred", action='store_true')
    parser.add_argument("--pred-mode", type=str, default="ex_pred", choices=["ex_pred", "true_ex"])
    args = parser.parse_args()
    main(args)
