import matplotlib.pyplot as plt
import numpy as np

def plot_diff_box(diff_dict, output_file):
    """
    Create a box plot comparing the value distributions of 'self' and 'cross' blocks.

    Parameters
    ----------
    diff_dict : Dict[str, Dict[int, torch.Tensor | np.ndarray | Sequence]]
        Dictionary with keys 'self' and 'cross', each mapping to a dict of block indices to arrays.
    output_file : str or pathlib.Path
        Path to save the generated plot.
    """
    data = []
    labels = []
    for key in ['self', 'cross']:
        if key in diff_dict:
            # Flatten and concatenate all arrays for this key
            arrays = []
            for arr in diff_dict[key].values():
                arr = arr.cpu().numpy()
                arr = arr[arr <= 100]
                arrays.append(arr.reshape(-1))
            if arrays:
                all_flat = np.concatenate(arrays)
                data.append(all_flat)
                labels.append(f"{key}-attention")
    if not data:
        raise ValueError("No data found in diff_dict for 'self' or 'cross'.")

    plt.figure(figsize=(6, 6))
    plt.boxplot(data, labels=labels, showmeans=False, meanline=True)
    plt.xlabel("Type", fontsize=16)
    plt.ylabel("Value distribution", fontsize=16)
    plt.title("Difference of true K and approximated K", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    