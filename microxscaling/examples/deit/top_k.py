import torch

def top_k_attn(attn, k):
    """
    Get the top k attention values and their indices.
    """
    top_k_attn = torch.topk(attn, k, dim=-1)
    top_k_attn = top_k_attn.values
    return top_k_attn

def top_k_pruning_with_softmax(attn, k):
    """
    Prune the attention values to keep only the top k values per row and apply softmax only on these values.
    Uses vectorized operations for better performance.
    
    Args:
        attn: attention tensor of shape (batch, head, N, N)
        k: number of top values to keep per row
    
    Returns:
        Pruned attention tensor with same shape as input, where only top k values have softmax applied
    """
    # Get top k values and indices
    top_k_values, top_k_indices = torch.topk(attn, k, dim=-1)
    
    # Apply softmax only to the top k values
    top_k_softmax = torch.softmax(top_k_values, dim=-1)
    
    # Create output tensor filled with zeros
    pruned_attn = torch.zeros_like(attn)
    
    # Create indices for scattering
    batch_size, num_heads, seq_len, _ = attn.shape
    batch_indices = torch.arange(batch_size).view(-1, 1, 1, 1).expand(-1, num_heads, seq_len, k)
    head_indices = torch.arange(num_heads).view(1, -1, 1, 1).expand(batch_size, -1, seq_len, k)
    seq_indices = torch.arange(seq_len).view(1, 1, -1, 1).expand(batch_size, num_heads, -1, k)
    
    # Scatter the softmaxed top-k values back to their original positions
    pruned_attn[batch_indices, head_indices, seq_indices, top_k_indices] = top_k_softmax
    
    return pruned_attn

def top_k_pruning(attn, k):
    """
    Legacy function maintained for compatibility.
    Now just calls top_k_pruning_with_softmax.
    """
    return top_k_pruning_with_softmax(attn, k)

