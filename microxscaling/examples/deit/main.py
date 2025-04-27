# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import torch.nn as nn
from pathlib import Path
import os

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
from augment import new_data_aug_generator

import models
import models_v2

import utils

## added by ckj
from mx.quantize import quantize_bfloat
from mx.elemwise_ops import quantize_elemwise_op
from mx.mx_ops import quantize_mx_op, _reshape_to_blocks, _shared_exponents
from mx import Linear, matmul
from top_k import classtopk
from exponent_based_prediction import exponent_approximation
## added by ckj to implement mx quantization

# def int_to_twos_complement(value, bits):
#     """Convert an integer to its two's complement binary representation."""
#     if value < 0:
#         value = (1 << bits) + value
#     return format(value, f'0{bits}b')

# def get_bits_from_format(format_name):
#     """Convert format name to number of bits."""
#     if format_name.startswith('int'):
#         return int(format_name[3:])
#     elif format_name.startswith('uint'):
#         return int(format_name[4:])
#     else:
#         raise ValueError(f"Unsupported format: {format_name}")

class QuantizedAttention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, orig_attn, mx_specs=None, top_k=True, k=20, exponent_based_prediction=True):
        super().__init__()
        self.mx_specs = mx_specs
        self.top_k = top_k
        self.k = k
        self.exponent_based_prediction = exponent_based_prediction
        self.num_heads = orig_attn.num_heads
        self.scale = orig_attn.scale
        self.top_k_obj = classtopk(k)
        self.exponent_based_obj =  None
        # mx-quantized linear layers
        self.qkv = Linear(
            orig_attn.qkv.in_features,
            orig_attn.qkv.out_features,
            bias=orig_attn.qkv.bias is not None,
            mx_specs=mx_specs
        )
        # mx-quantized linear layers
        self.proj = Linear(
            orig_attn.proj.in_features,
            orig_attn.proj.out_features,
            bias=orig_attn.proj.bias is not None,
            mx_specs=mx_specs
        )
        # inference dropout = 0
        self.proj_drop = nn.Dropout(orig_attn.proj_drop.p)
        self.attn_drop = nn.Dropout(orig_attn.attn_drop.p)

        # weights and biases
        self.qkv.weight.data = orig_attn.qkv.weight.data
        if orig_attn.qkv.bias is not None:
            self.qkv.bias.data = orig_attn.qkv.bias.data
            
        self.proj.weight.data = orig_attn.proj.weight.data
        if orig_attn.proj.bias is not None:
            self.proj.bias.data = orig_attn.proj.bias.data
            
        # Store last quantized values for inspection
        self.last_quantized_q = None
        self.last_quantized_k = None
        self.last_quantized_v = None
        self.last_quantized_attn = None
        self.origin_attn_values = None
        self.top_k_pruned_attn = None
        self.exponent_based_prediction_attn = None
        self.ex_quant_q = None
        self.ex_quant_k = None
        self.exponent_based_prediction_mask = None
        self.ori_pred_mask = None
        self.origin_softmax = None
        self.top_k_mismatch = None
        self.chosen_ori_indices = None
        self.chosen_ex_indices = None
        self.ori_top_k_softmax = None
        self.ex_top_k_softmax = None
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        k_t = k.transpose(-2, -1)
        
        # Quantize q, k, v to bfloat16 before matmul
        bf_q = quantize_elemwise_op(q, mx_specs=self.mx_specs, round=self.mx_specs["round_output"])
        bf_k = quantize_elemwise_op(k_t, mx_specs=self.mx_specs, round=self.mx_specs["round_output"])
        
        # Store quantized values
        self.last_quantized_q = quantize_mx_op(
            bf_q,
            self.mx_specs,
            elem_format=self.mx_specs["a_elem_format"],
            axes=[-1],
            round=self.mx_specs["round_mx_output"],
            predict_phase=False
        )
        self.last_quantized_k = quantize_mx_op(
            bf_k,
            self.mx_specs,
            elem_format=self.mx_specs["a_elem_format"],
            axes=[-2],
            round=self.mx_specs["round_mx_output"],
            predict_phase=False
        )
        
        # Use matmul with quantized inputs

        attn = matmul(q, k_t, mx_specs=self.mx_specs, mode_config='aa')
        self.origin_attn_values = attn
        origin_softmax = torch.softmax(attn, dim=-1)
        self.origin_softmax = origin_softmax


        # exponent-based top-k prediction
        self.exponent_based_obj = exponent_approximation(Q=q, K=k, mx_specs=self.mx_specs)
        
        # ex_quant_q, ex_quant_k = self.exponent_based_obj.exponent_based_sign_leading_ones()
        # ex_quant_q, ex_quant_k = self.exponent_based_obj.exponent_based_sign_bias()
        # ex_quant_q, ex_quant_k = self.exponent_based_obj.exponent_based_sign_top_2()
        # ex_quant_q, ex_quant_k = self.exponent_based_obj.exponent_based_hybrid_head_selector()
        ex_quant_q, ex_quant_k = self.exponent_based_obj.exponent_based_sign()
        # ex_quant_q, ex_quant_k = self.exponent_based_obj.exponent_based_threshold_exponent()
        self.ex_quant_q = ex_quant_q
        self.ex_quant_k = ex_quant_k
        ex_quant_k_t = ex_quant_k.transpose(-2, -1)

        pred_attn = ex_quant_q @ ex_quant_k_t
        pred_softmax = torch.softmax(pred_attn, dim=-1)
        self.exponent_based_prediction_attn = pred_softmax

        ex_pred_mask = self.top_k_obj.generate_top_k_mask(pred_attn)
        self.exponent_based_prediction_mask = ex_pred_mask

        ## original top-k prediction
        ori_pred_mask = self.top_k_obj.generate_top_k_mask(attn)
        self.ori_pred_mask = ori_pred_mask

        if self.exponent_based_prediction:
            mask = ex_pred_mask
        else:
            mask = ori_pred_mask
        
        ## top-k mismatch
        chosen_ori_indices = self.top_k_obj.top_k_indices(ori_pred_mask, k=self.k)
        chosen_ex_indices = self.top_k_obj.top_k_indices(ex_pred_mask, k=self.k)
        self.chosen_ori_indices = chosen_ori_indices
        self.chosen_ex_indices = chosen_ex_indices
        
        ori_top_k_softmax = self.top_k_obj.top_k_softmax(attn, ori_pred_mask, self.k)
        self.ori_top_k_softmax = ori_top_k_softmax

        ex_top_k_softmax = self.top_k_obj.top_k_softmax(attn, ex_pred_mask, self.k)
        self.ex_top_k_softmax = ex_top_k_softmax

        # Apply top-k pruning and softmax in one step
        if self.top_k:
            attn = self.top_k_obj.apply_masked_softmax(attn, mask)
            self.top_k_pruned_attn = attn       # inspect the top-k pruned attention
        else:
            attn = torch.softmax(attn, dim=-1)
        
        attn = self.attn_drop(attn)     # inference dropout = 0 


        bf_v = quantize_elemwise_op(v, mx_specs=self.mx_specs, round=self.mx_specs["round_output"])
        self.last_quantized_v = quantize_mx_op(
            bf_v,
            self.mx_specs,
            elem_format=self.mx_specs["a_elem_format"],
            axes=[-2],
            round=self.mx_specs["round_mx_output"],
            predict_phase=False,
        )
        bf_attn = quantize_elemwise_op(attn, mx_specs=self.mx_specs, round=self.mx_specs["round_output"])
        self.last_quantized_attn = quantize_mx_op(
            bf_attn,
            self.mx_specs,
            elem_format=self.mx_specs["a_elem_format"],
            axes=[-1],
            round=self.mx_specs["round_mx_output"],
            predict_phase=False,
        )

        x = matmul(attn, v, mx_specs=self.mx_specs, mode_config='aa')
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class QuantizedMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, orig_mlp, mx_specs=None):
        super().__init__()
        # Copy attributes from original MLP
        self.mx_specs = mx_specs
        self.act = orig_mlp.act
        self.drop = nn.Dropout(orig_mlp.drop.p)

        self.fc1 = Linear(
            orig_mlp.fc1.in_features,
            orig_mlp.fc1.out_features,
            bias=orig_mlp.fc1.bias is not None,
            mx_specs=mx_specs
        )
        
        self.fc2 = Linear(
            orig_mlp.fc2.in_features, 
            orig_mlp.fc2.out_features,
            bias=orig_mlp.fc2.bias is not None,
            mx_specs=mx_specs
        )
        
        # Copy weights and biases
        self.fc1.weight.data = orig_mlp.fc1.weight.data
        self.fc2.weight.data = orig_mlp.fc2.weight.data
        if orig_mlp.fc1.bias is not None:
            self.fc1.bias.data = orig_mlp.fc1.bias.data
        if orig_mlp.fc2.bias is not None:
            self.fc2.bias.data = orig_mlp.fc2.bias.data

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class QuantizedBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, orig_block, mx_specs=None, top_k=True, k=20, exponent_based_prediction=False):
        super().__init__()
        # Copy attributes from original block
        self.mx_specs = mx_specs
        self.norm1 = orig_block.norm1
        self.norm2 = orig_block.norm2
        self.drop_path = orig_block.drop_path
        
        # Replace with quantized versions
        self.attn = QuantizedAttention(
            orig_attn=orig_block.attn,
            mx_specs=mx_specs,
            top_k=top_k,
            k=k,
            exponent_based_prediction=exponent_based_prediction
        )
        
        # Create quantized MLP
        self.mlp = QuantizedMlp(
            orig_mlp=orig_block.mlp,
            mx_specs=mx_specs
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class SaveStats:
    def __init__(self, first_eval=False):
        self.first_eval = first_eval
        self.output_dir = 'outputs/quantization_stats'
        self.hooks = []  # Store hooks so we can remove them later
        self.has_run = False  # Track if we've already run once
    def int_to_twos_complement(self, value, bits):
        """Convert an integer to its two's complement binary representation."""
        if value < 0:
            value = (1 << bits) + value
        return format(value, f'0{bits}b')

    def get_bits_from_format(self, format_name):
        """Convert format name to number of bits."""
        if format_name.startswith('int'):
            return int(format_name[3:])
        elif format_name.startswith('uint'):
            return int(format_name[4:])
        else:
            raise ValueError(f"Unsupported format: {format_name}")
    def save_binary_stats(self, tensor, name, mx_specs, axes=[-1]):
        """Save shared exponents and binary representations of values.
        
        Args:
            tensor: Input tensor
            name: Name for the output file
            mx_specs: Quantization specifications
            axes: Axes along which to compute shared exponents
        """
        print(f"Saving binary stats for {name}")
        # Create directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Convert tensor to CPU if it's on GPU
        tensor = tensor.detach().cpu()
        
        # First reshape to blocks
        if mx_specs['block_size'] > 0:
            # print(f"Processing {name}, tensor shape: {tensor.shape}")
            tensor_blocked, reshaped_axes, orig_shape, padded_shape = _reshape_to_blocks(
                tensor, axes, mx_specs['block_size']
            )
            print(f"{name} blocked shape: {tensor_blocked.shape}")
            # Adjust axes for reshaped tensor
            shared_exp_axes = [x + 1 for x in reshaped_axes]
        else:
            tensor_blocked = tensor
            shared_exp_axes = axes
            
        # Get shared exponents
        shared_exp = _shared_exponents(
            tensor_blocked,
            method=mx_specs.get('shared_exp_method', 'max'),
            axes=shared_exp_axes,
            ebits=0
        )
        
        # Process in chunks to save memory
        with open(f'{self.output_dir}/{name}_binary.txt', 'w') as f:
            # Get the number of dimensions
            n_dims = len(tensor_blocked.shape)
            
            # Handle different tensor shapes
            if n_dims == 3:  # 3D tensor
                self._process_3d_tensor(tensor_blocked, shared_exp, f, mx_specs)
            elif n_dims == 4:  # 4D tensor (e.g., for attention)
                self._process_4d_tensor(tensor_blocked, shared_exp, f, mx_specs)
            elif n_dims == 5:  # 5D tensor (e.g., for attention)
                self._process_5d_tensor(tensor_blocked, shared_exp, f, mx_specs)
            else:
                raise ValueError(f"Unsupported tensor dimension: {n_dims}")
            
    def _process_3d_tensor(self, tensor_blocked, shared_exp, f, mx_specs):
        """Process a 3D tensor."""
        for dim1_idx in range(tensor_blocked.shape[0]):
            f.write(f"Token Index {dim1_idx}\n")
            for block_idx in range(tensor_blocked.shape[1]):
                f.write(f"  Block {block_idx}\n")
                block_values = tensor_blocked[dim1_idx, block_idx]
                exp = shared_exp[dim1_idx, block_idx].item()
                self._write_block_data(block_values, exp, f, mx_specs, indent="    ")

    def _process_4d_tensor(self, tensor_blocked, shared_exp, f, mx_specs):
        """Process a 4D tensor."""
        tensor_blocked = tensor_blocked[0]      # first batch
        shared_exp = shared_exp[0]
        for dim1_idx in range(tensor_blocked.shape[0]):
            f.write(f"Token Index {dim1_idx}\n")
            for block_idx in range(tensor_blocked.shape[1]):
                f.write(f"  Block {block_idx}\n")
                block_values = tensor_blocked[dim1_idx, block_idx]
                exp = shared_exp[dim1_idx, block_idx].item()
                self._write_block_data(block_values, exp, f, mx_specs, indent="    ")

    def _process_4d_orig_attn(self, tensor, f, mx_specs):
        """Process a 4D tensor."""
        tensor = tensor[0]      # first batch
        for dim1_idx in range(tensor.shape[0]):
            f.write(f"Head Index {dim1_idx}\n")
            for token_idx in range(tensor.shape[1]):
                f.write(f"  Token Index {token_idx}\n")
                for k_idx in range(tensor.shape[2]):
                    f.write(f"    {tensor[dim1_idx, token_idx, k_idx]}")
                f.write("\n")
    def _process_3d_top_k_mismatch(self, mismatch, f, mx_specs):
        """Process a 3D tensor."""
        mismatch = mismatch[0]
        for dim1_idx in range(mismatch.shape[0]):
            f.write(f"Head Index {dim1_idx}\n")
            for block_idx in range(mismatch.shape[1]):
                f.write(f"  Token Index {block_idx}: sum = {mismatch[dim1_idx, block_idx]}\n")
                

    def _process_5d_tensor(self, tensor_blocked, shared_exp, f, mx_specs):
        """Process a 5D tensor."""
        tensor_blocked = tensor_blocked[0]      # first batch
        shared_exp = shared_exp[0]
        for dim1_idx in range(tensor_blocked.shape[0]):
            f.write(f" Head Index {dim1_idx}\n")
            for dim2_idx in range(tensor_blocked.shape[1]):
                f.write(f"  Token Index {dim2_idx}\n")
                for block_idx in range(tensor_blocked.shape[2]):
                    f.write(f"    Block {block_idx}\n")
                    f.write(f"    {tensor_blocked[dim1_idx, dim2_idx, block_idx]}")
                    f.write("\n")
                    # block_values = tensor_blocked[dim1_idx, dim2_idx, block_idx]
                    # exp = shared_exp[dim1_idx, dim2_idx, block_idx].item()
                    # self._write_block_data(block_values, exp, f, mx_specs, indent="      ")
                    
    def _write_block_data(self, block_values, exp, f, mx_specs, indent="  "):
        """Write the data for a single block."""
        f.write(f"{indent}Shared Exponent: {exp}\n")
        f.write(f"{indent}Values: ")
        
        # Process values in smaller chunks
        values = block_values.reshape(-1)
        # Get number of bits from format name
        n_bits = self.get_bits_from_format(mx_specs['a_elem_format'])
        
        for i in range(0, len(values)):
            # Convert to quantized value
            val_bits = (values[i].item() / (2**exp)) * 2**(n_bits-2)
            val_bits = self.int_to_twos_complement(int(val_bits), n_bits)
            f.write(f"{val_bits} ")
        f.write("\n\n")

    def save_attn_stats(self, name, tensor, mx_specs):
        """Save attention stats."""
        with open(f'{self.output_dir}/{name}_binary.txt', 'w') as f:
            self._process_4d_orig_attn(tensor, f, mx_specs)
    def save_mismatch_stats(self, name, mismatch, mx_specs):
        """Save mismatch stats."""
        with open(f'{self.output_dir}/{name}_binary.txt', 'w') as f:
            self._process_3d_top_k_mismatch(mismatch, f, mx_specs)

    def save_quantization_stats(self, name, module, inp, out):
        """Hook function to save quantized activations and weights.""" 
        # Skip if this isn't the first evaluation or if we've already run
        if not self.first_eval or self.has_run:
            return
        # Create directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # For QuantizedAttention, save q, k, v values and quantized weights
        if isinstance(module, QuantizedAttention):
            # Get quantized weights and move to CPU
            quantized_qkv_weight = module.qkv.get_quantized_weight().detach().cpu()
            self.save_binary_stats(quantized_qkv_weight, f"{name}_qkv_weight", module.qkv.mx_specs)
            
            quantized_proj_weight = module.proj.get_quantized_weight().detach().cpu()
            self.save_binary_stats(quantized_proj_weight, f"{name}_proj_weight", module.proj.mx_specs)

            # Save quantized q, k, v values if available
            if module.last_quantized_q is not None:
                # print(module.last_quantized_q.size())
                self.save_binary_stats(module.last_quantized_q.detach().cpu(), f"{name}_q", module.mx_specs)
            if module.last_quantized_k is not None:
                # print(module.last_quantized_k.size())
                self.save_binary_stats(module.last_quantized_k.transpose(-2, -1).detach().cpu(), f"{name}_k", module.mx_specs)
            if module.ex_quant_q is not None:
                self.save_binary_stats(module.ex_quant_q.detach().cpu(), f"{name}_ex_q", module.mx_specs)
            if module.ex_quant_k is not None:
                self.save_binary_stats(module.ex_quant_k.detach().cpu(), f"{name}_ex_k", module.mx_specs)
            if module.last_quantized_v is not None:
                # print(module.last_quantized_v.size())
                self.save_binary_stats(module.last_quantized_v.transpose(-2, -1).detach().cpu(), f"{name}_v", module.mx_specs)
            if module.last_quantized_attn is not None:
                print(module.last_quantized_attn.size())
                self.save_binary_stats(module.last_quantized_attn.detach().cpu(), f"{name}_attn", module.mx_specs)
            # if module.origin_attn_values is not None:
            #     print(module.origin_attn_values.size())
            #     self.save_attn_stats(f"{name}_origin_attn_values", module.origin_attn_values.detach().cpu(), module.mx_specs)
            # if module.top_k_pruned_attn is not None:
            #     print(module.top_k_pruned_attn.size())
            #     self.save_attn_stats(f"{name}_top_k_pruned_attn", module.top_k_pruned_attn.detach().cpu(), module.mx_specs)
            # if module.exponent_based_prediction_attn is not None:
            #     print(module.exponent_based_prediction_attn.size())
            #     self.save_attn_stats(f"{name}_exponent_based_prediction_attn", module.exponent_based_prediction_attn.detach().cpu(), module.mx_specs)
            # if module.exponent_based_prediction_mask is not None:
            #     print(module.exponent_based_prediction_mask.size())
            #     self.save_attn_stats(f"{name}_exponent_based_prediction_mask", module.exponent_based_prediction_mask.detach().cpu(), module.mx_specs)
            # if module.ori_pred_mask is not None:
            #     print(module.ori_pred_mask.size())
            #     self.save_attn_stats(f"{name}_ori_pred_mask", module.ori_pred_mask.detach().cpu(), module.mx_specs)
            # if module.origin_softmax is not None:
            #     print(module.origin_softmax.size())
            #     self.save_attn_stats(f"{name}_origin_softmax", module.origin_softmax.detach().cpu(), module.mx_specs)
            if module.chosen_ori_indices is not None and module.chosen_ex_indices is not None:
                self.save_attn_stats(f"{name}_chosen_ori_indices", module.chosen_ori_indices.detach().cpu(), module.mx_specs)
                self.save_attn_stats(f"{name}_chosen_ex_indices", module.chosen_ex_indices.detach().cpu(), module.mx_specs)
            if module.ori_top_k_softmax is not None and module.ex_top_k_softmax is not None:
                self.save_attn_stats(f"{name}_ori_top_k_softmax", module.ori_top_k_softmax.detach().cpu(), module.mx_specs)
                self.save_attn_stats(f"{name}_ex_top_k_softmax", module.ex_top_k_softmax.detach().cpu(), module.mx_specs)
        # For QuantizedMlp, save activation and quantized weights
        elif isinstance(module, QuantizedMlp):
            # Save binary representations for fc1 and fc2 weights
            quantized_fc1_weight = module.fc1.get_quantized_weight()
            quantized_fc2_weight = module.fc2.get_quantized_weight()
            self.save_binary_stats(quantized_fc1_weight, f"{name}_fc1_weight", module.fc1.mx_specs)
            self.save_binary_stats(quantized_fc2_weight, f"{name}_fc2_weight", module.fc2.mx_specs)
        
        self.has_run = True
        del quantized_qkv_weight
        del quantized_proj_weight
        torch.cuda.empty_cache()

    def register_stats_hooks(self, model):
        """Register hooks on quantized attention and MLP layers in the model."""
        if not self.first_eval:
            print("Not first evaluation, skipping hook registration")
            return []
            
        if self.has_run:
            print("Already collected stats once, skipping hook registration")
            return []

        print("Registering hooks for first evaluation...")
        
        # Create closure to capture block_idx properly
        def make_hook(idx, layer_type='attn'):
            def hook(module, inp, out):
                self.save_quantization_stats(f'blocks_{idx}_{layer_type}', module, inp, out)
            return hook
        
        # Hook quantized layers in transformer blocks
        for block_idx, block in enumerate(model.blocks):

            # Hook attention if it's quantized
            if self.first_eval and block_idx == 0:
                # print(f"Hooking attention in block {block_idx}")
                if isinstance(block.attn, QuantizedAttention):
                    hook = block.attn.register_forward_hook(make_hook(block_idx, 'attn'))
                    self.hooks.append(hook)
                    
                if isinstance(block.mlp, QuantizedMlp):
                    hook = block.mlp.register_forward_hook(make_hook(block_idx, 'mlp'))
                    self.hooks.append(hook)
        
        print("Finished hooking quantized layers.")
        return self.hooks
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def apply_quantization_to_deit(model, config, first_eval=False, top_k=True, k=20, exponent_based_prediction=False):
    """
    Apply weight and activation quantization to specific parts of DeiT model
    
    Args:
        model: DeiT model instance
        config: dict containing quantization configuration
        first_eval (bool): Whether this is the first evaluation run
    """
    block_indices = config.get('blocks', [])
    components = config.get('components', ['attn', 'ffn'])
    mx_specs = config.get('mx_specs', {
            'w_elem_format': 'int4',
            'a_elem_format': 'int8',
            'scale_bits': 8,
            'block_size': 32,
            'bfloat': 16,
            'fp': 0,
            'round': 'floor',
            'round_mx_output': 'floor',
            'round_output': 'floor',
            'round_weight': 'floor',
            'custom_cuda': False,
            'quantize_backprop': False,
    })
    
    print(f"Applying quantization to blocks: {block_indices}")
    print(f"Quantizing components: {components}")
    
    # Get the device of the original model
    device = next(model.parameters()).device
    
    for idx in block_indices:
        if idx >= len(model.blocks):
            print(f"Block {idx} out of range, skipping")
            continue
            
        block = model.blocks[idx]
        if 'attn' in components:
            print(f"Quantizing attention in block {idx}")
            block.attn = QuantizedAttention(
                orig_attn=block.attn, 
                mx_specs=mx_specs,
                top_k=top_k,
                k=k,
                exponent_based_prediction=exponent_based_prediction
            ).to(device)
        
        if 'ffn' in components:
            print(f"Quantizing FFN in block {idx}")
            block.mlp = QuantizedMlp(
                orig_mlp=block.mlp,
                mx_specs=mx_specs
            ).to(device)
    
    # Register hooks to collect quantization statistics
    save_stats = SaveStats(first_eval)
    hooks = save_stats.register_stats_hooks(model)
    return model, hooks

# Function to verify quantization was applied correctly
def verify_quantization(model, block_indices):
    print("\nVerifying quantization:")
    for i, block in enumerate(model.blocks):
        is_quantized = i in block_indices
        attn_type = type(block.attn).__name__
        ffn_type = type(block.mlp).__name__
        
        status = "QUANTIZED" if is_quantized else "not quantized"
        print(f"Block {i}: {status}")
        print(f"  Attention: {attn_type}")
        print(f"  FFN: {ffn_type}")

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    
    # * Cosub params
    parser.add_argument('--cosub', action='store_true') 
    
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Quantization signal
    parser.add_argument('--quantize', action='store_true')
    # top k signal
    parser.add_argument('--top_k', action='store_true')
    # k signal
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--exponent_based_prediction', action='store_true')
    return parser
 
            
           


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)
        
    if args.attn_only:
        for name_p,p in model.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            print('no position encoding')
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print('no patch embed')
    
    # Load checkpoint if resume path is provided
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        model_without_ddp = model
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            lr_scheduler.step(args.start_epoch)
    
         
    
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    ## Apply quantization after loading the checkpoint
    quantization_config = {
        'blocks': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'components': ['attn', 'ffn'],
        'mx_specs': {
            'w_elem_format': 'int8',
            'a_elem_format': 'int8',
            'scale_bits': 8,
            'shared_exp_method': 'max',
            'block_size': 16,
            'bfloat': 16,
            'fp': 0,
            'bfloat_subnorms': True,
            'round': 'floor',
            'round_mx_output': 'floor',
            'round_output': 'floor',
            'round_weight': 'floor',
            'mx_flush_fp32_subnorms': False,
            'custom_cuda': False,
            'quantize_backprop': False,
        }
    }
    
    # First move model to device
    model.to(device)
    ## Eval with MXINT quantization--------------------------------------------------------------    
    output_dir = Path(args.output_dir)
    if args.eval:
        # Initialize SaveStats with first_eval flag
        save_stats = None
        if args.quantize:
            save_stats = SaveStats(first_eval=True)
            model, hooks = apply_quantization_to_deit(model, quantization_config, first_eval=True, top_k=args.top_k, k=args.k, exponent_based_prediction=args.exponent_based_prediction)
        
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
        # Clean up hooks and save stats instance
        if save_stats:
            save_stats.remove_hooks()
            del save_stats
        
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
            args = args,
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
             

        # test_stats = evaluate(data_loader_val, model, device)
        # print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
        # if max_accuracy < test_stats["acc1"]:
        #     max_accuracy = test_stats["acc1"]
        #     if args.output_dir:
        #         checkpoint_paths = [output_dir / 'best_checkpoint.pth']
        #         for checkpoint_path in checkpoint_paths:
        #             utils.save_on_master({
        #                 'model': model_without_ddp.state_dict(),
        #                 'optimizer': optimizer.state_dict(),
        #                 'lr_scheduler': lr_scheduler.state_dict(),
        #                 'epoch': epoch,
        #                 'model_ema': get_state_dict(model_ema),
        #                 'scaler': loss_scaler.state_dict(),
        #                 'args': args,
        #             }, checkpoint_path)
            
        # print(f'Max accuracy: {max_accuracy:.2f}%')

        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #              **{f'test_{k}': v for k, v in test_stats.items()},
        #              'epoch': epoch,
        #              'n_parameters': n_parameters}
        
        
        
        
        # if args.output_dir and utils.is_main_process():
        #     with (output_dir / "log.txt").open("a") as f:
        #         f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # Clean up hooks at the end
    if hooks:
        for hook in hooks:
            hook.remove()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
