import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from mx.elemwise_ops import quantize_elemwise_op
from mx.mx_ops import quantize_mx_op, _shared_exponents, _reshape_to_blocks, _undo_reshape_to_blocks

from funcs.utils import write_data

class exponent_approximation:
    def __init__(self, Q, K, mx_specs):   
        self.mx_specs = mx_specs
        self.Q = Q
        self.K = K
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
    
        

    def exponent_based_sign(self):
        """
        Q, K approximation (proposed exp-sign):

        Q = shared exponent, element INT8 -> +1 or -1
        K = shared exponent, element INT8 -> +1 or -1
        """
        
        # print(self.shared_exponent_Q.shape)

        # Convert all non-zero values to +1 or -1 based on their sign
        signs_Q = torch.where(self.reshaped_MX_Q < 0, -1, +1)
        signs_K = torch.where(self.reshaped_MX_K < 0, -1, +1)
        # Expand shared_exponents to match the block size dimension

        ##################################################
        ## Generate patterns for compute power analysis ##
        ##################################################
        
        # signs_file_path_Q = "signs_Q.txt"
        # signs_file_path_K = "signs_K.txt"
        # shared_exponents_file_path_Q = "shared_exponents_Q.txt"
        # shared_exponents_file_path_K = "shared_exponents_K.txt"
        # if not os.path.exists(signs_file_path_Q) or not os.path.exists(signs_file_path_K) or not os.path.exists(shared_exponents_file_path_Q) or not os.path.exists(shared_exponents_file_path_K):
        #     data_Q = torch.where(signs_Q == -1, 1, 0)[0][0].flatten().cpu()
        #     data_K = torch.where(signs_K == -1, 1, 0)[0][0].flatten().cpu()
        #     shared_exponents_Q = self.shared_exponent_Q[0][0].flatten().cpu().int()
        #     shared_exponents_K = self.shared_exponent_K[0][0].flatten().cpu().int()
        #     if not os.path.exists(signs_file_path_Q):
        #         write_data(data_Q, signs_file_path_Q)
        #     if not os.path.exists(signs_file_path_K):
        #         write_data(data_K, signs_file_path_K)
        #     if not os.path.exists(shared_exponents_file_path_Q):
        #         write_data(shared_exponents_Q, shared_exponents_file_path_Q)
        #     if not os.path.exists(shared_exponents_file_path_K):
        #         write_data(shared_exponents_K, shared_exponents_file_path_K)
        # expanded_exponents_Q = self.shared_exponent_Q.expand_as(self.reshaped_MX_Q)
        # expanded_exponents_K = self.shared_exponent_K.expand_as(self.reshaped_MX_K)


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

    def two_step_leading_ones(self):
        """
        Q, K approximation (EXION):

        Q = shared exponent, (element INT8 -> two-step leading ones)
        K = shared exponent, (element INT8 -> two-step leading ones)
        """

        signs_Q = torch.sign(self.reshaped_MX_Q)
        signs_K = torch.sign(self.reshaped_MX_K)
        expanded_shared_exponent_Q = self.shared_exponent_Q.expand_as(self.reshaped_MX_Q)
        expanded_shared_exponent_K = self.shared_exponent_K.expand_as(self.reshaped_MX_K)

        # Extract raw int8 values by dividing by 2^shared_exponent
        raw_int8_Q = (self.reshaped_MX_Q / (2 ** expanded_shared_exponent_Q)) * 64
        raw_int8_K = (self.reshaped_MX_K / (2 ** expanded_shared_exponent_K)) * 64

        first_leading_ones_Q = _shared_exponents(torch.abs(raw_int8_Q), method="none", axes=[-1], ebits=0)
        first_leading_ones_K = _shared_exponents(torch.abs(raw_int8_K), method="none", axes=[-1], ebits=0)
        temp_Q = torch.where(raw_int8_Q-(2 ** first_leading_ones_Q) < 0, 0, raw_int8_Q-(2 ** first_leading_ones_Q))
        temp_K = torch.where(raw_int8_K-(2 ** first_leading_ones_K) < 0, 0, raw_int8_K-(2 ** first_leading_ones_K))

        second_leading_ones_Q = _shared_exponents(temp_Q, method="none", axes=[-1], ebits=0)
        second_leading_ones_K = _shared_exponents(temp_K, method="none", axes=[-1], ebits=0)

        expanded_first_leading_ones_Q = first_leading_ones_Q.expand_as(self.reshaped_MX_Q)
        expanded_first_leading_ones_K = first_leading_ones_K.expand_as(self.reshaped_MX_K)
        expanded_second_leading_ones_Q = second_leading_ones_Q.expand_as(self.reshaped_MX_Q)
        expanded_second_leading_ones_K = second_leading_ones_K.expand_as(self.reshaped_MX_K)

        approx_Q = signs_Q * expanded_shared_exponent_Q * ((2 ** expanded_first_leading_ones_Q) + (2 ** expanded_second_leading_ones_Q)) / 64
        approx_K = signs_K * expanded_shared_exponent_K * ((2 ** expanded_first_leading_ones_K) + (2 ** expanded_second_leading_ones_K)) / 64
        # Convert all non-zero values to +- leading ones of each element
        
        approx_Q = _undo_reshape_to_blocks(approx_Q, self.padded_shape_Q, self.orig_shape_Q, self.axes_Q)
        approx_K = _undo_reshape_to_blocks(approx_K, self.padded_shape_K, self.orig_shape_K, self.axes_K)


        ##################################################
        ## Generate patterns for compute power analysis ##
        ##################################################

        # # Only write files if they don't already exist
        # sgn_file_path_Q = "sgn_Q.txt"
        # sgn_file_path_K = "sgn_K.txt"
        # exp0_file_path_Q = "exp0_Q.txt"
        # exp1_file_path_Q = "exp1_Q.txt"
        # exp0_file_path_K = "exp0_K.txt"
        # exp1_file_path_K = "exp1_K.txt"
        # shared_exp_Q = "shared_exp_Q.txt"
        # shared_exp_K = "shared_exp_K.txt"
        # if not os.path.exists(sgn_file_path_Q) or not os.path.exists(sgn_file_path_K) or not os.path.exists(exp0_file_path_Q) or not os.path.exists(exp0_file_path_K) or not os.path.exists(exp1_file_path_Q) or not os.path.exists(exp1_file_path_K) or not os.path.exists(shared_exp_Q) or not os.path.exists(shared_exp_K):
        #     sgn_data_Q = torch.where(signs_Q == -1, 1, 0)[0][0].flatten().cpu().int()
        #     sgn_data_K = torch.where(signs_K == -1, 1, 0)[0][0].flatten().cpu().int()
        #     exp0_data_Q = torch.where(expanded_first_leading_ones_Q[0][0] < 0, 8, expanded_first_leading_ones_Q[0][0]).flatten().cpu().int()
        #     exp0_data_K = torch.where(expanded_first_leading_ones_K[0][0] < 0, 8, expanded_first_leading_ones_K[0][0]).flatten().cpu().int()
        #     exp1_data_Q = torch.where(expanded_second_leading_ones_Q[0][0] < 0, 8, expanded_second_leading_ones_Q[0][0]).flatten().cpu().int()
        #     exp1_data_K = torch.where(expanded_second_leading_ones_K[0][0] < 0, 8, expanded_second_leading_ones_K[0][0]).flatten().cpu().int()
        #     shared_exp_data_Q = self.shared_exponent_Q[0][0].flatten().cpu().int()
        #     shared_exp_data_K = self.shared_exponent_K[0][0].flatten().cpu().int()
            
        #     if not os.path.exists(sgn_file_path_Q):
        #         write_data(sgn_data_Q, sgn_file_path_Q)
        #     if not os.path.exists(sgn_file_path_K):
        #         write_data(sgn_data_K, sgn_file_path_K)
        #     if not os.path.exists(exp0_file_path_Q) or not os.path.exists(exp0_file_path_K) or not os.path.exists(exp1_file_path_Q) or not os.path.exists(exp1_file_path_K) or not os.path.exists(shared_exp_Q) or not os.path.exists(shared_exp_K):
        #         write_data(exp0_data_Q, exp0_file_path_Q)
        #     if not os.path.exists(exp0_file_path_K):
        #         write_data(exp0_data_K, exp0_file_path_K)
        #     if not os.path.exists(exp1_file_path_Q):
        #         write_data(exp1_data_Q, exp1_file_path_Q)
        #     if not os.path.exists(exp1_file_path_K):
        #         write_data(exp1_data_K, exp1_file_path_K)
        #     if not os.path.exists(shared_exp_Q):
        #         write_data(shared_exp_data_Q, shared_exp_Q)
        #     if not os.path.exists(shared_exp_K):
        #         write_data(shared_exp_data_K, shared_exp_K)

        del signs_Q, signs_K, first_leading_ones_Q, first_leading_ones_K, second_leading_ones_Q, second_leading_ones_K, temp_Q, temp_K
        torch.cuda.empty_cache()

        return approx_Q, approx_K
    
    def MXINT4(self):
        """
        Q, K approximation (Sanger):

        Q = MXINT4
        K = MXINT4
        """
        
        approx_Q = quantize_mx_op(
            quantize_elemwise_op(self.Q, self.mx_specs, round=self.mx_specs["round_output"]),
            self.mx_specs, 
            elem_format="int4", 
            axes=[-1],
            round=self.mx_specs["round_mx_output"], 
            predict_phase=False)
        approx_K = quantize_mx_op(
            quantize_elemwise_op(self.K, self.mx_specs, round=self.mx_specs["round_output"]),
            self.mx_specs, 
            elem_format="int4", 
            axes=[-1],
            round=self.mx_specs["round_mx_output"], 
            predict_phase=False)
        
        # Extract int4 values from the quantized tensors
        # First, we need to reshape to blocks to extract the int4 values
        reshaped_approx_Q, axes_Q, orig_shape_Q, padded_shape_Q = _reshape_to_blocks(approx_Q, [-1], self.mx_specs["block_size"])
        reshaped_approx_K, axes_K, orig_shape_K, padded_shape_K = _reshape_to_blocks(approx_K, [-1], self.mx_specs["block_size"])
        
        # Get shared exponents for the int4 quantized data
        shared_exp_Q = _shared_exponents(reshaped_approx_Q, method=self.shared_exponent_method, axes=[-1], ebits=0)
        shared_exp_K = _shared_exponents(reshaped_approx_K, method=self.shared_exponent_method, axes=[-1], ebits=0)
        
        ##################################################
        ## Generate patterns for compute power analysis ##
        ##################################################

        # # Only write files if they don't already exist
        # mxint4_file_path_Q = "mxint4_Q.txt"
        # mxint4_file_path_K = "mxint4_K.txt"
        # mxint4_shared_exp_Q = "mxint4_shared_exp_Q.txt"
        # mxint4_shared_exp_K = "mxint4_shared_exp_K.txt"
        
        # if not os.path.exists(mxint4_file_path_Q) or not os.path.exists(mxint4_file_path_K) or not os.path.exists(mxint4_shared_exp_Q) or not os.path.exists(mxint4_shared_exp_K):
        #     # Extract raw int4 values by dividing by shared exponents
        #     # reshaped_approx contains: int4_value * 2^shared_exponent
        #     # To get int4_value: divide by 2^shared_exponent
            
        #     # Expand shared exponents to match tensor dimensions
        #     expanded_shared_exp_Q = shared_exp_Q.expand_as(reshaped_approx_Q)
        #     expanded_shared_exp_K = shared_exp_K.expand_as(reshaped_approx_K)
            
        #     # Extract raw int4 values by dividing by 2^shared_exponent
        #     raw_int4_Q = (reshaped_approx_Q / (2 ** expanded_shared_exp_Q)) * 4
        #     raw_int4_K = (reshaped_approx_K / (2 ** expanded_shared_exp_K)) * 4

        #     # Extract data from first batch, first head and convert to int
        #     int4_data_Q = raw_int4_Q[0][0].flatten().cpu().int()
        #     int4_data_K = raw_int4_K[0][0].flatten().cpu().int()
        #     shared_exp_data_Q = shared_exp_Q[0][0].flatten().cpu().int()
        #     shared_exp_data_K = shared_exp_K[0][0].flatten().cpu().int()
            
        #     if not os.path.exists(mxint4_file_path_Q):
        #         write_data(int4_data_Q, mxint4_file_path_Q)
        #     if not os.path.exists(mxint4_file_path_K):
        #         write_data(int4_data_K, mxint4_file_path_K)
        #     if not os.path.exists(mxint4_shared_exp_Q):
        #         write_data(shared_exp_data_Q, mxint4_shared_exp_Q)
        #     if not os.path.exists(mxint4_shared_exp_K):
        #         write_data(shared_exp_data_K, mxint4_shared_exp_K)

        # mxint8_file_path_Q = "mxint8_Q.txt"
        # mxint8_file_path_K = "mxint8_K.txt"
        # mxint8_file_shared_exp_Q = "mxint8_shared_exp_Q.txt"
        # mxint8_file_shared_exp_K = "mxint8_shared_exp_K.txt"
        
        # if not os.path.exists(mxint8_file_path_Q) or not os.path.exists(mxint8_file_path_K) or not os.path.exists(mxint8_file_shared_exp_Q) or not os.path.exists(mxint8_file_shared_exp_K):
        #     expanded_shared_exp_Q = self.shared_exponent_Q.expand_as(self.reshaped_MX_Q)
        #     expanded_shared_exp_K = self.shared_exponent_K.expand_as(self.reshaped_MX_K)
        #     mxint8_data_Q = (self.reshaped_MX_Q / (2 ** expanded_shared_exp_Q))[0][0] * 64
        #     mxint8_data_K = (self.reshaped_MX_K / (2 ** expanded_shared_exp_K))[0][0] * 64
        #     mxint8_shared_exp_Q = self.shared_exponent_Q[0][0]
        #     mxint8_shared_exp_K = self.shared_exponent_K[0][0]
            
        #     if not os.path.exists(mxint8_file_path_Q):
        #         write_data(mxint8_data_Q.flatten().cpu().int(), mxint8_file_path_Q)
        #     if not os.path.exists(mxint8_file_path_K):
        #         write_data(mxint8_data_K.flatten().cpu().int(), mxint8_file_path_K)
        #     if not os.path.exists(mxint8_file_shared_exp_Q):
        #         write_data(mxint8_shared_exp_Q.flatten().cpu().int(), mxint8_file_shared_exp_Q)
        #     if not os.path.exists(mxint8_file_shared_exp_K):
        #         write_data(mxint8_shared_exp_K.flatten().cpu().int(), mxint8_file_shared_exp_K)
        del self.Q, self.K
        torch.cuda.empty_cache()
        return approx_Q, approx_K
    
    def partial_K(self):
        """
        Q, K approximation:

        Q = proposed exp-sign approximation
        K = MXINT8
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
        """
        Q, K approximation:

        Q = MXINT8
        K = proposed exp-sign approximation
        """
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


        
        
        