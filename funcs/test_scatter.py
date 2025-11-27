import sys
import os
# Add the microxscaling directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'microxscaling'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from mx.elemwise_ops import quantize_elemwise_op
from mx.mx_ops import quantize_mx_op, _shared_exponents, _reshape_to_blocks, _undo_reshape_to_blocks

# ==== toy data ====
np.random.seed(0)
torch.manual_seed(0)
num_vecs = 200
d = 128
B = 32
T = d // B

# Create numpy arrays first, then convert to torch tensors
Q_np = np.random.randn(num_vecs, d)
K_np = np.random.randn(num_vecs, d)

# Convert to PyTorch tensors (float32 by default)
Q = torch.from_numpy(Q_np).float()
K = torch.from_numpy(K_np).float()

mx_specs = {
    'w_elem_format': 'int8',
    'a_elem_format': 'int8',
    'scale_bits': 8,
    'shared_exp_method': 'max',
    'block_size': 32,
    'bfloat': 32,
    'fp': 0,
    'bfloat_subnorms': True,
    'round': 'nearest',
    'round_mx_output': 'nearest',
    'round_output': 'nearest',
    'round_weight': 'nearest',
    'mx_flush_fp32_subnorms': False,
    'custom_cuda': False,
    'quantize_backprop': False,
}

MX_Q = quantize_mx_op(
        quantize_elemwise_op(Q, mx_specs, round=mx_specs["round_output"]),
        mx_specs, 
        elem_format=mx_specs["a_elem_format"], 
        axes=[-1],
        round=mx_specs["round_mx_output"], 
        predict_phase=False)
MX_K = quantize_mx_op(
        quantize_elemwise_op(K, mx_specs, round=mx_specs["round_output"]),
        mx_specs, 
        elem_format=mx_specs["a_elem_format"], 
        axes=[-1],
        round=mx_specs["round_mx_output"], 
        predict_phase=False)
        
Z = MX_Q @ MX_K.T

reshaped_MX_Q, axes_Q, orig_shape_Q, padded_shape_Q = _reshape_to_blocks(MX_Q, [-1], mx_specs["block_size"])
reshaped_MX_K, axes_K, orig_shape_K, padded_shape_K = _reshape_to_blocks(MX_K, [-1], mx_specs["block_size"])
shared_exponent_Q = _shared_exponents(reshaped_MX_Q, method=mx_specs["shared_exp_method"], axes=[-1], ebits=0)
shared_exponent_K = _shared_exponents(reshaped_MX_K, method=mx_specs["shared_exp_method"], axes=[-1], ebits=0)

print(reshaped_MX_Q.shape, axes_Q, orig_shape_Q, padded_shape_Q)
print(reshaped_MX_K.shape, axes_K, orig_shape_K, padded_shape_K)
print(shared_exponent_Q.shape, shared_exponent_K.shape)
print(Z.shape)

def plot_scatter(x, y, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(5,5))
    plt.scatter(x, y, s=2, alpha=0.3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# Convert tensors back to numpy for the experiments
Q_np = Q.detach().cpu().numpy()
K_np = K.detach().cpu().numpy()
MX_Q_np = MX_Q.detach().cpu().numpy()
MX_K_np = MX_K.detach().cpu().numpy()
Z_np = Z.detach().cpu().numpy()
reshaped_MX_Q_np = reshaped_MX_Q.detach().cpu().numpy()
reshaped_MX_K_np = reshaped_MX_K.detach().cpu().numpy()
shared_exponent_Q_np = shared_exponent_Q.detach().cpu().numpy()
shared_exponent_K_np = shared_exponent_K.detach().cpu().numpy()

# ==== Experiment 1 ====
intra_exp, intra_true = [], []
# Iterate over each vector and each block
for i in range(num_vecs):  # For each vector
    for t in range(T):  # For each block (T=16 blocks now with B=8)
        # Get the block data for vector i, block t
        q_block = reshaped_MX_Q_np[i, t, :]  # Shape: (8,)
        k_block = reshaped_MX_K_np[i, t, :]  # Shape: (8,)
        
        # Calculate dot product for this block
        z_t = np.dot(q_block, k_block)  # Scalar
        
        # Get shared exponents for this block
        exp_q_t = shared_exponent_Q_np[i, t, 0]  # Scalar
        exp_k_t = shared_exponent_K_np[i, t, 0]  # Scalar

        # Calculate exponent approximation (restore the 2** operation)
        xi_t = 2 ** (exp_q_t + exp_k_t)  # Scalar
        
        intra_true.append(z_t)
        intra_exp.append(xi_t)

intra_true = np.array(intra_true)
intra_exp = np.array(intra_exp)

# Remove the debug prints
print(f"intra_true range: [{np.min(intra_true):.3f}, {np.max(intra_true):.3f}]")
print(f"intra_exp range: [{np.min(intra_exp):.3f}, {np.max(intra_exp):.3f}]")

# Use absolute values and handle zeros/negatives properly
abs_intra_true = np.abs(intra_true)
# Add small epsilon to avoid log(0)
abs_intra_true = np.maximum(abs_intra_true, 1e-10)

# Use log for better visualization and correlation
log_intra_true = np.log(abs_intra_true)
log_intra_exp = np.log(np.maximum(intra_exp, 1e-10))

rho1, _ = spearmanr(log_intra_exp, log_intra_true)
plot_scatter(log_intra_exp, log_intra_true, f"Exp vs intra-block |z_t| (Spearman ρ={rho1:.2f})", "log S_exp (block)", "log |z_t|", "exp_intra_block.png")

print(f"Experiment 1 - Intra-block correlation: {rho1:.3f}")

# ==== Experiment 2 ====
# Calculate sum of exponent approximations across all blocks for each vector pair
S_exp = np.zeros((num_vecs, num_vecs))
for i in range(num_vecs):
    for j in range(num_vecs):
        exp_sum = 0
        for t in range(T):
            exp_q_t = shared_exponent_Q_np[i, t, 0]
            exp_k_t = shared_exponent_K_np[j, t, 0]
            exp_sum += 2 ** (exp_q_t + exp_k_t)
        S_exp[i, j] = exp_sum

rho2, _ = spearmanr(S_exp.ravel(), np.abs(Z_np.ravel()))
plot_scatter(np.log1p(S_exp.ravel()), np.log1p(np.abs(Z_np.ravel())), f"Exp-only vs inter-block |Z| (Spearman ρ={rho2:.2f})", "log Σ S_exp", "log |Z|", "exp_inter_block.png")

print(f"Experiment 2 - Exp-only correlation: {rho2:.3f}")

# ==== Experiment 3 ====
# Calculate sign-based correlations
Q_blocks = np.where(reshaped_MX_Q_np < 0, -1, 1)  # Shape: (num_vecs, T, B)
K_blocks = np.where(reshaped_MX_K_np < 0, -1, 1)  # Shape: (num_vecs, T, B)

# Method 1: Loop-based approach (clearer and matches experimental design)
S_meta = np.zeros((num_vecs, num_vecs))
for i in range(num_vecs):
    for j in range(num_vecs):
        meta_sum = 0
        for t in range(T):
            # Calculate sign correlation for this block
            sign_results = np.dot(Q_blocks[i, t, :], K_blocks[j, t, :])
            
            # Get exponent approximation
            exp_q_t = shared_exponent_Q_np[i, t, 0]
            exp_k_t = shared_exponent_K_np[j, t, 0]
            xi = 2 ** (exp_q_t + exp_k_t)
            
            meta_sum += xi * sign_results
        S_meta[i, j] = meta_sum

# Add debug info to understand the values
print(f"S_meta range: [{np.min(S_meta):.3f}, {np.max(S_meta):.3f}]")
print(f"S_meta zeros: {np.sum(S_meta == 0)}/{S_meta.size}")
print(f"S_meta negatives: {np.sum(S_meta < 0)}/{S_meta.size}")

print(Z.shape, S_meta.shape)
rho3, _ = spearmanr(S_meta.ravel(), Z_np.ravel())  # Use Z_np directly, not abs

# Handle negative values properly for plotting
# Use signed log: sign(x) * log(1 + |x|) for better visualization
def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))

S_meta_log = (S_meta.ravel())
Z_log = (Z_np.ravel())

plot_scatter(S_meta_log, Z_log, f"Exp+Sign vs inter-block Z (Spearman ρ={rho3:.3f})", "signed log S_meta", "signed log Z", "exp_sign_inter_block.png")

print(f"Experiment 3 - Exp+Sign correlation: {rho3:.3f}")
