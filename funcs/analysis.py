import numpy as np
import os
import torch
import re, ast
from pathlib import Path

# class diff_analysis:
#     def __init__(self, diff_dict):
#         self.diff_dict = diff_dict
    
#     def collect_block_diff(self, array, block_idx, type:str="all"):
#        if type not in self.diff_dict:
#            self.diff_dict[type] = {}
#        self.diff_dict[type][block_idx] = array

def create_file(output_file):
    dir_name = os.path.dirname(output_file)
    if dir_name:  # Only make directories if dir_name is not empty
        os.makedirs(dir_name, exist_ok=True)
    with open(output_file, "w") as f:
        pass
    
def save_idx_file(idx, output_file, block_idx:int=None):
    B, H, N, T = idx.shape
    with open(output_file, "a") as f:
        f.write(f"Cross-attention Block {block_idx}\n")
        for h in range(H):
            f.write(f" Head {h}:\n")
            for n in range(N):
                f.write(f"  Token {n:3d}: {(idx[1, h, n, :].tolist())}\n")        # sorted(idx[1, h, n, :].tolist())

def save_diff_score_file(diff_score, output_file, block_idx:int=None):
    with open(output_file, "a") as f:
        f.write(f"{diff_score}\n")

def init_analysis_files(attn_type:str, anal_dir, k, approx_flag, pred_mode, total_timestep):
    file_name_dict = {}
    anal_dir = anal_dir + f'/{attn_type}'
    anal_dir = f"{anal_dir}/{pred_mode}" if approx_flag else f"{anal_dir}/true"
    print(f"approx_flag mode: {approx_flag}")
    print(f"Initializing {anal_dir} ...")
    print(f"Creating files ...")
    for timestep in range(total_timestep):
        file_name_dict[timestep] = {}
        file_name_dict[timestep]['idx'] = f"{anal_dir}/top{k}_idx_t{timestep}.txt"
        file_name_dict[timestep]['vals'] = f"{anal_dir}/top{k}_vals_t{timestep}.txt"
        file_name_dict[timestep]['diff_idx'] = f"{anal_dir}/top{k}_diff_idx_t{timestep}.txt"
        # print(f"Creating file {file_name_dict[timestep]['idx']} ...")
        # create_file(file_name_dict[timestep]['idx'])
        # print(f"Creating file {file_name_dict[timestep]['vals']} ...")
        # create_file(file_name_dict[timestep]['vals'])
        # print(f"Creating file {file_name_dict[timestep]['diff_idx']} ...")
        create_file(file_name_dict[timestep]['diff_idx'])
    return file_name_dict

def total_chosen_k(pred_idx):
    """
    Calculate the total number of unique indices chosen across all attention rows.
    
    Example:
        Row 0 chooses indices: [0, 1, 2]
        Row 1 chooses indices: [0, 5, 6]
        Total unique indices chosen: {0, 1, 2, 5, 6} = 5 unique indices
    
    Args:
        pred_idx: Multi-dimensional tensor where:
                 - Last dimension contains the chosen indices
                 - Second-to-last dimension (-2) is the attention row number (e.g., 256)
                 - Other dimensions are batch, head, etc.
    
    Returns:
        dict: Dictionary with statistics averaged over batches and heads:
            - 'unique_k_count': Average number of unique indices chosen across all rows
            - 'coverage_rate': Average ratio of unique indices to total possible indices
    """
    batch_size, num_heads = pred_idx.shape[0], pred_idx.shape[1]
    results = []
    
    # Process each batch and head separately
    for b in range(batch_size):
        for h in range(num_heads):
            # Get indices for this batch and head across all 256 rows
            indices_2d = pred_idx[b, h]  # Shape: [256, k]
            
            # print(f"Indices 2D: {indices_2d}")
            # Flatten and get unique indices (pred_idx already contains chosen indices)
            all_indices = indices_2d.flatten()
            unique_indices = torch.unique(all_indices)
            
            # Count unique indices
            unique_k_count = len(unique_indices)
            
            # Calculate coverage rate (assuming indices range from 0 to some max value)
            # max_possible_idx = torch.max(all_indices).item() if len(all_indices) > 0 else 1
            coverage_rate = unique_k_count / pred_idx.shape[-2]
            
            results.append({
                'unique_k_count': unique_k_count,
                'coverage_rate': coverage_rate
            })
    
    # Average across all batches and heads
    avg_unique_k = sum(r['unique_k_count'] for r in results) / len(results)
    avg_coverage_rate = sum(r['coverage_rate'] for r in results) / len(results)
    
    # return {
    #     'unique_k_count': avg_unique_k,
    #     'coverage_rate': avg_coverage_rate
    # }
    return avg_coverage_rate

def parse_tokens(path, token_re):
    """Return a dict {token_id: list} from lines that look like:
       '  Token   3: [10, 55, ...]'"""
    tokens = {}
    with path.open() as f:
        block_idx = 0
        head_idx = 0
        for line in f:
            token = token_re.search(line)
            if token:
                token_id = int(token.group(2))
                if block_idx not in tokens:
                    tokens[block_idx] = {}
                if head_idx not in tokens[block_idx]:
                    tokens[block_idx][head_idx] = {}
                tokens[block_idx][head_idx][token_id] = ast.literal_eval(token.group(3))
                if token_id == 255 and head_idx == 15:
                    head_idx = 0
                    block_idx += 1
                elif token_id == 255:
                    head_idx += 1
                # print(f"Token {token_id}: {tokens[token_id]}")
    return tokens

def diff_idx_analysis(true_idx: torch.Tensor, pred_idx: torch.Tensor):
    """
    For each row (all leading dims except the last), compute the sum of the
    indices that appear in `true_idx` but NOT in the corresponding row of
    `pred_idx`.

    Output:
        diff_ratio_all: float
        The ratio of the sum of the matching scores for each token, average over all heads (16 heads) and tokens (256 tokens)
        Each block and each timestep has a diff_ratio_all
    """
    # Mask for elements that also appear in pred_idx (True = present in pred_idx)
    present_mask = torch.isin(true_idx, pred_idx)
    unique_vals = torch.where(present_mask, true_idx, torch.zeros_like(true_idx))
    true_sum = true_idx.sum(dim=-1, keepdim=True)
    diff_sum = unique_vals.sum(dim=-1, keepdim=True)
    diff_ratio = diff_sum / true_sum
    # diff_ratio = true_sum
    diff_ratio_all = diff_ratio[0:100,:,:,0].sum().item()       # item(): convert tensor to float
    all_dim = 100 * diff_ratio.shape[1] * diff_ratio.shape[2]
    diff_ratio_all = diff_ratio_all / all_dim
    return diff_ratio_all

def mismatch_analysis(true_top20_file, pred_top60_file):
    true_top20_file = Path(true_top20_file)
    pred_top60_file = Path(pred_top60_file)
    token_re = re.compile(r'(Token\s+(\d+):\s*)(\[[^\]]*\])')

    true_top20_tokens = parse_tokens(true_top20_file, token_re)
    pred_top60_tokens = parse_tokens(pred_top60_file, token_re)
    # print(true_top20_tokens)
    # print(pred_top60_tokens)
    diff_lines = []
    with true_top20_file.open() as src:
        block_idx = 0
        head_idx = 0
        for idx, line in enumerate(src):
            token = token_re.search(line)
            if token:
                prefix = token.group(1)
                token_id = int(token.group(2))
                true_tokens = true_top20_tokens[block_idx][head_idx][token_id]
                pred_tokens = pred_top60_tokens[block_idx][head_idx][token_id]
                diff_tokens = [x for x in true_tokens if x not in pred_tokens]
                diff_lines.append(f"{prefix}{len(diff_tokens)}: {diff_tokens}\n")
                if token_id == 255 and head_idx == 15:
                    head_idx = 0
                    block_idx += 1
                elif token_id == 255:
                    head_idx += 1
            else:
                diff_lines.append(line)
    out_file = Path("mismatch_idx.txt")
    out_file.write_text("".join(diff_lines))
    print(f"Diff file written to: {out_file}")
    return out_file

if __name__ == "__main__":
    base_dir = "/work/tttpd9bjo/diffusion/DiT/DiT-XL-2-256x256"
    true_top20_idx = f"{base_dir}/analysis/self_attention/true/"
    pred_top60_idx = f"{base_dir}/analysis/self_attention/ex_pred/"
    true_top20_name = "top154_vals_t0.txt"
    pred_top60_name = "top154_vals_t0.txt"
    true_top20_file = true_top20_idx + true_top20_name
    pred_top60_file = pred_top60_idx + pred_top60_name
    mismatch_analysis(true_top20_file, pred_top60_file)