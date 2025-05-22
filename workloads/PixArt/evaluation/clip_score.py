import torch
from PIL import Image
from torchmetrics.multimodal.clip_score import CLIPScore
from pathlib import Path
from torchvision import transforms as T
import argparse

def main(args):
    prompts = Path(args.prompt_pth).read_text().splitlines()
    img_dir = Path(args.img_pth)  # your folder of PNG/JPGs

    # Initialize CLIPScore with the official CLIP model
    metric = CLIPScore(
        model_name_or_path="openai/clip-vit-large-patch14",
    ).to("cuda")  # Move to GPU after creation
    # Create transform to convert PIL image to tensor
    transform = T.ToTensor()
    
    scores = []
    for idx, prompt in enumerate(prompts):
        # Load and convert image to RGB
        image = Image.open(img_dir / f"{idx}.jpg").convert("RGB")
        image_tensor = transform(image).to("cuda")
        # Calculate CLIP score - using keyword arguments
        with torch.inference_mode():
            score = metric(source=image_tensor, target=prompt)  # Use keyword arguments
            scores.append(score.item())
            
    print(f"total scores: {scores}")
    print(f"CLIPScore: {sum(scores) / len(scores)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-pth", type=str, default="../evaluation/generated_images/alpha-512/top512/true_exp/")
    parser.add_argument("--prompt-pth", type=str, default="../prompts/sample.txt")
    args = parser.parse_args()
    main(args) 