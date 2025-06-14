import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image


def create_npz_from_sample_folder(sample_dir, npz_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{npz_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def main(args):
    create_npz_from_sample_folder(
        sample_dir=args.sample_dir,
        npz_dir=args.npz_dir,
        num=args.num
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str, default="/work/tttpd9bjo/diffusion/DiT/DiT-XL-2-256x256/samples")
    parser.add_argument("--npz_dir", type=str, default="/work/tttpd9bjo/diffusion/DiT/DiT-XL-2-256x256/samples/npz")
    parser.add_argument("--num", type=int, default=15000)
    args = parser.parse_args()
    main(args)