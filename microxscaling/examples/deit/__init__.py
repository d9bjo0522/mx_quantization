import sys
import os

# Add the microxscaling root directory to the Python path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import necessary modules
from mx.quantize import quantize_bfloat
