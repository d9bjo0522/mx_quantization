import sys
# add mx quantization root directory to sys.path (for exponent_based_prediction.py)
root_dir = "/home/tttpd9bjo/mx_quantization"
mx_dir = root_dir + "/microxscaling"

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if mx_dir not in sys.path:
    sys.path.insert(0, mx_dir)

from .exponent_based_prediction import exponent_approximation