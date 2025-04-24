import torch
import numpy as np

class classtopk:
    def __init__(self, k):
        self.k = k

    def top_k_attn(self, attn):
        """
        Get the top k attention values and their indices.
        """
        top_k_attn = torch.topk(attn, self.k, dim=-1)
        top_k_attn = top_k_attn.values
        return top_k_attn

    def generate_top_k_mask(self, attn):
        """
        Generate a binary mask for top k attention values.
        
        Args:
            attn: attention tensor of shape (batch, head, N, N)
            k: number of top values to keep per row
        
        Returns:
            Binary mask tensor of same shape as input, where 1 indicates top-k positions
        """
        # Get top k indices
        _, top_k_indices = torch.topk(attn, self.k, dim=-1)
        
        # Create binary mask tensor
        mask = torch.zeros_like(attn, dtype=torch.bool)
        
        # Create indices for setting mask values
        batch_size, num_heads, seq_len, _ = attn.shape
        batch_indices = torch.arange(batch_size, device=attn.device).view(-1, 1, 1, 1).expand(-1, num_heads, seq_len, self.k)
        head_indices = torch.arange(num_heads, device=attn.device).view(1, -1, 1, 1).expand(batch_size, -1, seq_len, self.k)
        seq_indices = torch.arange(seq_len, device=attn.device).view(1, 1, -1, 1).expand(batch_size, num_heads, -1, self.k)
        
        # Set top-k positions to True
        mask[batch_indices, head_indices, seq_indices, top_k_indices] = True
        
        return mask
    
    def top_k_indices(self, mask, k: int = 20):
        """
        Parameters
        ----------
        mask : torch.Tensor or np.ndarray, bool
            An array of shape (..., N) whose last axis contains exactly `k` True
            values for every slice.
        k : int, default 20
            Number of True values expected on the last axis.

        Returns
        -------
        idx : torch.Tensor, int
            Same leading shape as `mask` but with the last axis length `k`.
            Each element along that axis is the position (0‑based) where `mask`
            is True.
        """
        device = mask.device  # Remember original device
            
        # 1.  Flatten every slice on the last axis into a 2‑D view
        flat = mask.reshape(-1, mask.shape[-1]) 
        
        # 2.  Collect column indices (= positions on the last axis) that are True
        #     torch.where returns (row_indices, col_indices); we only need the cols
        cols = torch.where(flat)[1]
        
        # 3.  Put the k indices belonging to the same slice back next to each other
        idx = cols.reshape(mask.shape[:-1] + (k,))

        # 4.  (optional) sort them so they appear in ascending order
        idx, _ = torch.sort(idx, dim=-1)
        
        # Return tensor on the original device
        return idx.to(device)
    
        
    def top_k_softmax(self, attn, mask, k):
        """
        Compute softmax values for the top k positions in each attention head.
        
        Args:
            attn: attention tensor of shape (batch, head, N, N)
            mask: boolean mask of same shape as attn
            k: number of top positions to consider
            
        Returns:
            Tensor of shape (batch, head, N, k) containing softmax values for top k positions
        """
        results = self.apply_masked_softmax(attn, mask)
        indices = self.top_k_indices(mask, k)
        results = results.gather(dim=-1, index=indices)
        
        return results
    
    def apply_masked_softmax(self, attn, mask):
        """
        Apply softmax only to the masked positions.
        
        Args:
            attn: attention tensor of shape (batch, head, N, N)
            mask: boolean mask of same shape as attn
        
        Returns:
            Attention tensor with softmax applied only to masked positions
        """
        # Create output tensor
        result = torch.zeros_like(attn)
        
        # Get masked values for softmax
        masked_attn = attn.clone()
        masked_attn[~mask] = float('-inf')  # Set non-masked values to -inf
        
        # Apply softmax
        result = torch.softmax(masked_attn, dim=-1)
    
        return result

