import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from mx.elemwise_ops import quantize_elemwise_op
from mx.mx_ops import quantize_mx_op, _shared_exponents, _reshape_to_blocks, _undo_reshape_to_blocks


def shared_exp_intra_block(reshaped_MX_Q, reshaped_MX_K, shared_exponent_Q, shared_exponent_K):
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