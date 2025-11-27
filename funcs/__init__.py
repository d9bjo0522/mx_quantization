import sys
# add mx quantization root directory to sys.path (for exponent_based_prediction.py)
root_dir = "/home/tttpd9bjo/mx_quantization"
mx_dir = root_dir + "/microxscaling"

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if mx_dir not in sys.path:
    sys.path.insert(0, mx_dir)

from .exponent_based_prediction import exponent_approximation
from .elsa_approximation import elsa_approximation, _create_structured_orthogonal_matrix, _modified_gram_schmidt
from .utils import write_data
from .analysis import save_idx_file, create_file, mismatch_analysis, init_analysis_files, diff_idx_analysis, save_diff_score_file, total_chosen_k
# from .plots import plot_diff_box