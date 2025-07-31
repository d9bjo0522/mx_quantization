import torch
import torch.nn as nn
import torch.nn.functional as F

from mx.elemwise_ops import quantize_elemwise_op
from mx.mx_ops import quantize_mx_op, _shared_exponents, _reshape_to_blocks, _undo_reshape_to_blocks

class exponent_approximation:
    def __init__(self, Q, K, mx_specs):   
        self.mx_specs = mx_specs
        # self.Q = Q
        # self.K = K
        # self.bf_Q = quantize_elemwise_op(self.Q, self.mx_specs, round=self.mx_specs["round_output"])
        # self.bf_K = quantize_elemwise_op(self.K, self.mx_specs, round=self.mx_specs["round_output"])
        self.MX_Q = quantize_mx_op(
            quantize_elemwise_op(Q, self.mx_specs, round=self.mx_specs["round_output"]),
            self.mx_specs, 
            elem_format=self.mx_specs["a_elem_format"], 
            axes=[-1],
            round=self.mx_specs["round_mx_output"], 
            predict_phase=False)
        self.MX_K = quantize_mx_op(
            quantize_elemwise_op(K, self.mx_specs, round=self.mx_specs["round_output"]),
            self.mx_specs, 
            elem_format=self.mx_specs["a_elem_format"], 
            axes=[-1],
            round=self.mx_specs["round_mx_output"], 
            predict_phase=False)
        self.shared_exponent_method = self.mx_specs.get("shared_exp_method", "max")
        self.reshaped_MX_Q, self.axes_Q, self.orig_shape_Q, self.padded_shape_Q = _reshape_to_blocks(self.MX_Q, [-1], self.mx_specs["block_size"])
        self.reshaped_MX_K, self.axes_K, self.orig_shape_K, self.padded_shape_K = _reshape_to_blocks(self.MX_K, [-1], self.mx_specs["block_size"])
        self.shared_exponent_Q = _shared_exponents(self.reshaped_MX_Q, method=self.shared_exponent_method, axes=[-1], ebits=0)
        self.shared_exponent_K = _shared_exponents(self.reshaped_MX_K, method=self.shared_exponent_method, axes=[-1], ebits=0)
        self.true_exponent_Q = _shared_exponents(self.reshaped_MX_Q, method="none", axes=[-1], ebits=0)
        self.true_exponent_K = _shared_exponents(self.reshaped_MX_K, method="none", axes=[-1], ebits=0)
        # del self.MX_Q, self.MX_K
        # torch.cuda.empty_cache()
    
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
    def get_second_largest_exponents(self, tensor):
        """
        Extract second largest exponents from tensor values.
        """
        abs_tensor = torch.abs(tensor)
        # print(abs_tensor.shape)
        first_largest, _ = torch.max(abs_tensor, dim=-1, keepdim=True)
        new_tensor = torch.where(abs_tensor == first_largest, 0, abs_tensor)
        second_largest, _ = torch.max(new_tensor, dim=-1, keepdim=True)
        second_largest = torch.floor(
            torch.log2(
                second_largest + 1e-10 * (second_largest == 0).type(second_largest.dtype)
            )
        )
        return second_largest
        

    def exponent_based_sign(self):
        """
        Approximates the exponent of the tensor using the max absolute value in each block.
        Each element in a block is converted to +1 or -1 based on its sign, then scaled by 2^exponent.
        """
        
        
        # Convert all non-zero values to +1 or -1 based on their sign
        signs_Q = torch.where(self.reshaped_MX_Q < 0, -1, +1)
        signs_K = torch.where(self.reshaped_MX_K < 0, -1, +1)
        # Expand shared_exponents to match the block size dimension

        expanded_exponents_Q = self.shared_exponent_Q.expand_as(self.reshaped_MX_Q)
        expanded_exponents_K = self.shared_exponent_K.expand_as(self.reshaped_MX_K)
        # expanded_exponents_Q = second_largest_exponents_Q.expand_as(self.reshaped_MX_Q)
        # expanded_exponents_K = second_largest_exponents_K.expand_as(self.reshaped_MX_K)
        # Scale the signs by 2^exponent
        approx_Q = signs_Q * (2 ** expanded_exponents_Q)
        approx_K = signs_K * (2 ** expanded_exponents_K)
        # Reshape back to original dimensions
        approx_Q = _undo_reshape_to_blocks(approx_Q, self.padded_shape_Q, self.orig_shape_Q, self.axes_Q)
        approx_K = _undo_reshape_to_blocks(approx_K, self.padded_shape_K, self.orig_shape_K, self.axes_K)
        
        ## free memory
        del signs_Q, signs_K, expanded_exponents_Q, expanded_exponents_K, self.reshaped_MX_Q, self.reshaped_MX_K, self.shared_exponent_Q, self.shared_exponent_K
        torch.cuda.empty_cache()
        return approx_Q, approx_K
    def exponent_based_sign_leading_ones(self):
        """
        Approximate each elements with the true leading ones.
        """
        signs_Q = torch.where(self.reshaped_MX_Q < 0, -1, +1)
        signs_K = torch.where(self.reshaped_MX_K < 0, -1, +1)
        true_exponent_Q = self.true_exponent_Q.expand_as(self.reshaped_MX_Q)
        true_exponent_K = self.true_exponent_K.expand_as(self.reshaped_MX_K)

        # Convert all non-zero values to +- leading ones of each element
        leading_ones_Q = signs_Q * (2 ** true_exponent_Q)
        leading_ones_K = signs_K * (2 ** true_exponent_K)

        approx_Q = _undo_reshape_to_blocks(leading_ones_Q, self.padded_shape_Q, self.orig_shape_Q, self.axes_Q)
        approx_K = _undo_reshape_to_blocks(leading_ones_K, self.padded_shape_K, self.orig_shape_K, self.axes_K)
        return approx_Q, approx_K
    
    def partial_K(self):
        """
        Approximates the exponent of the tensor using the max absolute value in each block.
        Each element in a block is converted to +1 or -1 based on its sign, then scaled by 2^exponent.
        """
        
        
        # Convert all non-zero values to +1 or -1 based on their sign
        signs_Q = torch.where(self.reshaped_MX_Q < 0, -1, +1)
        
        # Expand shared_exponents to match the block size dimension
        expanded_exponents_Q = self.shared_exponent_Q.expand_as(self.reshaped_MX_Q)
        

        # Scale the signs by 2^exponent

        ## partial K
        approx_Q = signs_Q * (2 ** expanded_exponents_Q)
        approx_K = self.reshaped_MX_K
        
        # Reshape back to original dimensions
        approx_Q = _undo_reshape_to_blocks(approx_Q, self.padded_shape_Q, self.orig_shape_Q, self.axes_Q)
        approx_K = _undo_reshape_to_blocks(approx_K, self.padded_shape_K, self.orig_shape_K, self.axes_K)

        return approx_Q, approx_K
    
    def partial_Q(self):
        signs_K = torch.where(self.reshaped_MX_K < 0, -1, +1)
        # Expand shared_exponents to match the block size dimension
        expanded_exponents_K = self.shared_exponent_K.expand_as(self.reshaped_MX_K)
        approx_Q = self.reshaped_MX_Q
        approx_K = signs_K * (2 ** expanded_exponents_K)
        # Reshape back to original dimensions
        approx_Q = _undo_reshape_to_blocks(approx_Q, self.padded_shape_Q, self.orig_shape_Q, self.axes_Q)
        approx_K = _undo_reshape_to_blocks(approx_K, self.padded_shape_K, self.orig_shape_K, self.axes_K)
        
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


        
        
        