import torch
from models import deit_tiny_patch16_224  # Import directly from local models.py

# Example code to print block structure
def print_block_structure(model):
    print("Model Structure:")
    for i, block in enumerate(model.blocks):
        print(f"\nBlock {i}:")
        for name, module in block.named_children():
            print(f"  {name}: {type(module).__name__}")

# Create DeiT-tiny model using our local implementation
model = deit_tiny_patch16_224(pretrained=True)
print_block_structure(model)

# Print additional model information
print("\nTotal number of blocks:", len(model.blocks))
print("Embedding dimension:", model.embed_dim)
print("Number of heads:", model.blocks[0].attn.num_heads)