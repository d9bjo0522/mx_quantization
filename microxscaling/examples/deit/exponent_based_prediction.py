import torch
import torch.nn as nn
import torch.nn.functional as F

from mx.elemwise_ops import quantize_elemwise_op
from mx.mx_ops import quantize_mx_op, _shared_exponents, _reshape_to_blocks, _undo_reshape_to_blocks

def most_frequent_exp_per_block(tensor):
    # Get the last dimension size
    last_dim = tensor.size(-1)
    
    # Reshape the tensor to 2D: (all_other_dims, last_dim)
    reshaped = tensor.reshape(-1, last_dim)
    
    # Sort each row
    sorted_values, _ = torch.sort(reshaped, dim=1)
    
    # Calculate differences between adjacent elements
    diffs = torch.diff(sorted_values, dim=1)
    
    # Add a column of ones at the end to handle boundary
    padding = torch.ones(reshaped.size(0), 1, device=tensor.device)
    diffs = torch.cat([diffs, padding], dim=1)
    
    # Find where values change (diff != 0)
    changes = (diffs != 0).float()
    
    # Calculate run lengths
    run_lengths = changes.cumsum(dim=1)
    
    # Get the value that appears most frequently in each row
    max_run_idx = run_lengths.max(dim=1)[1]
    most_frequent = torch.gather(sorted_values, 1, max_run_idx.unsqueeze(1)).squeeze(1)
    
    # Reshape back to original dimensions except last
    return most_frequent.view(tensor.size()[:-1])

def get_bits_from_format(format_name):
    """Convert format name to number of bits."""
    if format_name.startswith('int'):
        return int(format_name[3:])
    elif format_name.startswith('uint'):
        return int(format_name[4:])
    else:
        raise ValueError(f"Unsupported format: {format_name}")
    
class exponent_approximation:
    def __init__(self, Q, K, mx_specs):   
        self.mx_specs = mx_specs
        self.Q = Q
        self.K = K
        self.bf_Q = quantize_elemwise_op(self.Q, self.mx_specs, round=self.mx_specs["round_output"])
        self.bf_K = quantize_elemwise_op(self.K, self.mx_specs, round=self.mx_specs["round_output"])
        self.MX_Q = quantize_mx_op(self.bf_Q, self.mx_specs, elem_format=self.mx_specs["a_elem_format"], axes=[-1], round=self.mx_specs["round_mx_output"], predict_phase=True)
        self.MX_K = quantize_mx_op(self.bf_K, self.mx_specs, elem_format=self.mx_specs["a_elem_format"], axes=[-1], round=self.mx_specs["round_mx_output"], predict_phase=True)
        self.shared_exponent_method = self.mx_specs.get("shared_exp_method", "max")
        self.reshaped_MX_Q, self.axes_Q, self.orig_shape_Q, self.padded_shape_Q = _reshape_to_blocks(self.MX_Q, [-1], self.mx_specs["block_size"])
        self.reshaped_MX_K, self.axes_K, self.orig_shape_K, self.padded_shape_K = _reshape_to_blocks(self.MX_K, [-1], self.mx_specs["block_size"])
        self.shared_exponent_Q = _shared_exponents(self.reshaped_MX_Q, method=self.shared_exponent_method, axes=[-1], ebits=0)
        self.shared_exponent_K = _shared_exponents(self.reshaped_MX_K, method=self.shared_exponent_method, axes=[-1], ebits=0)
    def get_true_exponents(self, tensor):
        """
        Extract true exponents from tensor values.
        """
        # Handle zeros separately to avoid log2 of 0
        abs_tensor = torch.abs(tensor)
        mask = abs_tensor > 0
        result = torch.zeros_like(abs_tensor)
        
        # Calculate exponents for non-zero values
        result[mask] = torch.floor(torch.log2(abs_tensor[mask]))
        
        return result
    
    def most_frequent_exp_per_block(self, tensor):
        # Get the last dimension size
        last_dim = tensor.size(-1)
        
        # Reshape the tensor to 2D: (all_other_dims, last_dim)
        reshaped = tensor.reshape(-1, last_dim)
        
        # Get unique values and their counts per row
        most_frequent = []
        # Process all rows in parallel
        unique_vals, inverse_indices, counts = torch.unique(reshaped, dim=1, return_inverse=True, return_counts=True)
        
        # Get indices of max counts for each row
        max_count_indices = torch.argmax(counts)
        
        # Get the most common value for each row
        most_frequent = unique_vals[:, max_count_indices]
        # print(most_frequent.size())
        most_frequent = most_frequent.view(tensor.size()[:-1])
        # print(most_frequent.size())
        # most_frequent = most_frequent.unsqueeze(-1).expand_as(tensor)
        return most_frequent
    
    def exponent_based_sign(self):
        """
        Approximates the exponent of the tensor using the max absolute value in each block.
        Each element in a block is converted to +1 or -1 based on its sign, then scaled by 2^exponent.
        """
        
        
        # Convert all non-zero values to +1 or -1 based on their sign
        signs_Q = torch.where(self.reshaped_MX_Q < 0, -1, +1)
        signs_K = torch.where(self.reshaped_MX_K < 0, -1, +1)
        # signs_Q = torch.sign(self.reshaped_MX_Q)
        # signs_K = torch.sign(self.reshaped_MX_K)

        # Expand shared_exponents to match the block size dimension
        expanded_exponents_Q = self.shared_exponent_Q.expand_as(self.reshaped_MX_Q)
        expanded_exponents_K = self.shared_exponent_K.expand_as(self.reshaped_MX_K)

        # Scale the signs by 2^exponent
        approx_Q = signs_Q * (2 ** expanded_exponents_Q)
        approx_K = signs_K * (2 ** expanded_exponents_K)
        
        # Reshape back to original dimensions
        approx_Q = _undo_reshape_to_blocks(approx_Q, self.padded_shape_Q, self.orig_shape_Q, self.axes_Q)
        approx_K = _undo_reshape_to_blocks(approx_K, self.padded_shape_K, self.orig_shape_K, self.axes_K)
        return approx_Q, approx_K

    def exponent_based_sign_leading_ones(self):
        """
        Approximate each elements with the true leading ones.
        """

        true_exponent_Q = self.get_true_exponents(self.reshaped_MX_Q)
        true_exponent_K = self.get_true_exponents(self.reshaped_MX_K)

        # Convert all non-zero values to +- leading ones of each element
        leading_ones_Q = torch.sign(self.reshaped_MX_Q) * (2 ** true_exponent_Q)
        leading_ones_K = torch.sign(self.reshaped_MX_K) * (2 ** true_exponent_K)

        approx_Q = _undo_reshape_to_blocks(leading_ones_Q, self.padded_shape_Q, self.orig_shape_Q, self.axes_Q)
        approx_K = _undo_reshape_to_blocks(leading_ones_K, self.padded_shape_K, self.orig_shape_K, self.axes_K)
        return approx_Q, approx_K
    
    def exponent_based_sign_bias(self):
        # Get true exponents for Q and K
        true_exponent_Q = self.get_true_exponents(self.reshaped_MX_Q)
        true_exponent_K = self.get_true_exponents(self.reshaped_MX_K)

        # Get most frequent exponent per block
        most_freq_exp_Q = self.most_frequent_exp_per_block(true_exponent_Q)
        most_freq_exp_K = self.most_frequent_exp_per_block(true_exponent_K)
        # print(most_freq_exp_Q)
        # Expand most frequent exponents to match original shape
        expanded_exponents_Q = self.shared_exponent_Q.expand_as(self.reshaped_MX_Q)
        expanded_exponents_K = self.shared_exponent_K.expand_as(self.reshaped_MX_K)
        most_freq_exp_Q = most_freq_exp_Q.unsqueeze(-1).expand_as(self.reshaped_MX_Q)
        most_freq_exp_K = most_freq_exp_K.unsqueeze(-1).expand_as(self.reshaped_MX_K)
        # Get signs
        signs_Q = torch.sign(self.reshaped_MX_Q)
        signs_K = torch.sign(self.reshaped_MX_K)
        
        most_freq_exp_Q = torch.where(true_exponent_Q == expanded_exponents_Q, expanded_exponents_Q, most_freq_exp_Q)
        most_freq_exp_K = torch.where(true_exponent_K == expanded_exponents_K, expanded_exponents_K, most_freq_exp_K)
        # Create approximated values using most frequent exponents and signs
        approx_Q = signs_Q * (2.0 ** most_freq_exp_Q)
        approx_K = signs_K * (2.0 ** most_freq_exp_K)
        approx_Q = _undo_reshape_to_blocks(approx_Q, self.padded_shape_Q, self.orig_shape_Q, self.axes_Q)
        approx_K = _undo_reshape_to_blocks(approx_K, self.padded_shape_K, self.orig_shape_K, self.axes_K)
        return approx_Q, approx_K

    def get_top_2_exponents(self, tensor):
        """
        Get the top-2 exponents from the tensor in a vectorized way.
        Returns a tensor where each value is either the most frequent or second most frequent exponent.
        """
        exp_tensor = self.get_true_exponents(tensor)
        *leading_dims, last_dim = exp_tensor.shape
        
        # Reshape to 2D for processing
        reshaped = exp_tensor.reshape(-1, last_dim)
        # Sort descending to find largest unique numbers quickly
        sorted_vals, _ = torch.sort(reshaped, dim=1, descending=True)

        # Identify largest value per row
        first_vals = sorted_vals[:, 0]

        # Identify second-largest unique value per row efficiently
        not_equal_first = sorted_vals != first_vals.unsqueeze(1)
        has_second_val = not_equal_first.any(dim=1)

        # Use first value where no second-largest value exists
        second_vals = torch.where(
            has_second_val,
            sorted_vals.masked_fill(~not_equal_first, float('-inf')).max(dim=1).values,
            first_vals
        )

        # Replace elements: largest stay as largest, rest replaced by second-largest
        first_vals_expanded = first_vals.unsqueeze(1).expand_as(reshaped)
        second_vals_expanded = second_vals.unsqueeze(1).expand_as(reshaped)

        mask = reshaped == first_vals_expanded
        result = torch.where(mask, first_vals_expanded, second_vals_expanded)
       
        return result.view(*leading_dims, last_dim)

    def exponent_based_sign_top_2(self):
        """
        Approximate each elements with top-1 or top-2 exponents

        """
        top_2_exponents_Q = self.get_top_2_exponents(self.reshaped_MX_Q)
        top_2_exponents_K = self.get_top_2_exponents(self.reshaped_MX_K)

        # Convert all non-zero values to +- top-2 exponents of each element
        top_2_exponents_Q = torch.sign(self.reshaped_MX_Q) * (2 ** top_2_exponents_Q)
        top_2_exponents_K = torch.sign(self.reshaped_MX_K) * (2 ** top_2_exponents_K)

        approx_Q = _undo_reshape_to_blocks(top_2_exponents_Q, self.padded_shape_Q, self.orig_shape_Q, self.axes_Q)
        approx_K = _undo_reshape_to_blocks(top_2_exponents_K, self.padded_shape_K, self.orig_shape_K, self.axes_K)
        return approx_Q, approx_K
    
    def exponent_based_hybrid_head_selector(self):
        """
        Hybrid head selector based on exponent-based prediction.
        """
        # Get true exponents for Q and K
        batch_dim, head_dim, token_dim, block_dim, last_dim = self.reshaped_MX_Q.shape
        # Expand shared_exponents to match the block size dimension
        expanded_exponents_Q = self.shared_exponent_Q.expand_as(self.reshaped_MX_Q)
        expanded_exponents_K = self.shared_exponent_K.expand_as(self.reshaped_MX_K)
        true_exponent_Q = self.get_true_exponents(self.reshaped_MX_Q)
        true_exponent_K = self.get_true_exponents(self.reshaped_MX_K)

        
        # For first head use true exponents, others use expanded exponents
        sign_Q = torch.sign(self.reshaped_MX_Q)
        sign_K = torch.sign(self.reshaped_MX_K)

        hybrid_exponents_Q = torch.where(
            torch.arange(head_dim, device=true_exponent_Q.device).view(1, -1, 1, 1, 1) == 2,
            expanded_exponents_Q,
            true_exponent_Q
        )
        
        hybrid_exponents_K = torch.where(
            torch.arange(head_dim, device=true_exponent_K.device).view(1, -1, 1, 1, 1) == 2,
            expanded_exponents_K,
            true_exponent_K
        )

        # Convert to power of 2 and apply signs
        hybrid_Q = sign_Q * (2 ** hybrid_exponents_Q)
        hybrid_K = sign_K * (2 ** hybrid_exponents_K)

        # Reshape back to original dimensions
        approx_Q = _undo_reshape_to_blocks(hybrid_Q, self.padded_shape_Q, self.orig_shape_Q, self.axes_Q)
        approx_K = _undo_reshape_to_blocks(hybrid_K, self.padded_shape_K, self.orig_shape_K, self.axes_K)
        return approx_Q, approx_K
    
    def exponent_based_threshold_exponent(self):
        """
        Threshold the exponent of the tensor based on the true exponents.
        """
        
        expanded_exponents_Q = self.shared_exponent_Q.expand_as(self.reshaped_MX_Q)
        expanded_exponents_K = self.shared_exponent_K.expand_as(self.reshaped_MX_K)
        true_exponent_Q = self.get_true_exponents(self.reshaped_MX_Q)
        true_exponent_K = self.get_true_exponents(self.reshaped_MX_K)
        
        # threshold the exponent of the tensor based on the true exponents
        threshold_Q = torch.where(true_exponent_Q < expanded_exponents_Q-1, expanded_exponents_Q-1, true_exponent_Q)
        threshold_K = torch.where(true_exponent_K < expanded_exponents_K-1, expanded_exponents_K-1, true_exponent_K)

        # Convert to power of 2 and apply signs
        threshold_Q = torch.sign(self.reshaped_MX_Q) * (2 ** threshold_Q)
        threshold_K = torch.sign(self.reshaped_MX_K) * (2 ** threshold_K)

        approx_Q = _undo_reshape_to_blocks(threshold_Q, self.padded_shape_Q, self.orig_shape_Q, self.axes_Q)
        approx_K = _undo_reshape_to_blocks(threshold_K, self.padded_shape_K, self.orig_shape_K, self.axes_K)
        return approx_Q, approx_K


        
        
        