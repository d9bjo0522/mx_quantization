import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import os

def create_npz_from_sample_folder(sample_dir, npz_dir, file_name, resolution=256, num=5000):
    """
    Builds a single .npz file from a folder of image samples, resizing all to 256x256.
    """
    # Get all image files and filter by extension
    all_files = os.listdir(sample_dir)
    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.PNG', '.JPG', '.JPEG'))]
    
    # Sort files for consistent ordering
    image_files.sort()
    
    # Limit to num files
    print(f"Requested {num} images")
    print(f"Available {len(image_files)} images")
    if len(image_files) > num:
        image_files = image_files[:num]
    elif len(image_files) < num:
        raise ValueError(f"Not enough images: Only {len(image_files)} image files found, but {num} are requested.")
    else:
        print(f"Found {len(image_files)} image files to process")
    
    samples = []
    for i, file in enumerate(tqdm(image_files, desc="Loading and resizing images")):
        sample_pil = Image.open(os.path.join(sample_dir, file))
        
        # Convert to RGB if needed (handles RGBA, grayscale, etc.)
        if sample_pil.mode != 'RGB':
            sample_pil = sample_pil.convert('RGB')
        
        # Resize to 256x256
        sample_pil = sample_pil.resize((resolution, resolution), Image.LANCZOS)
        
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
        
        # Close the image to free memory
        sample_pil.close()
    
    if not samples:
        print("No valid images found!")
        return None
    
    print(f"Stacking {len(samples)} images...")
    samples = np.stack(samples)
    
    # All images should now be 256x256x3
    expected_shape = (len(samples), resolution, resolution, 3)
    assert samples.shape == expected_shape, f"Expected {expected_shape}, got {samples.shape}"
    
    print(f"Final shape: {samples.shape}")
    npz_path = f"{npz_dir}/{file_name}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def main(args):
    create_npz_from_sample_folder(
        sample_dir=args.sample_dir,
        npz_dir=args.npz_dir,
        file_name=args.file_name,
        num=args.num
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", type=str, default="/work/tttpd9bjo/diffusion/DiT/DiT-XL-2-256x256/samples")
    parser.add_argument("--npz_dir", type=str, default="/work/tttpd9bjo/diffusion/DiT/DiT-XL-2-256x256/samples/npz")
    parser.add_argument("--num", type=int, default=15000)
    parser.add_argument("--file_name", type=str, default="ImageNet256")
    args = parser.parse_args()
    main(args)