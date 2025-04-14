import sys
import os

# Get the parent directory of the current file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the parent directory to the system path
sys.path.insert(0, parent_dir)

from mx.specs import MxSpecs
from mx.mx_ops import quantize_mx_op
import torch
import numpy as np

class MXQuantizer(object):
    def __init__(self, elem_format, block_size, scale_bits):
        self.elem_format = elem_format
        self.block_size = block_size
        self.scale_bits = scale_bits
        self.my_specs = MxSpecs(a_elem_format=self.elem_format, w_elem_format=self.elem_format, block_size=self.block_size, scale_bits=self.scale_bits)
        
    def quantize(self, x):
        qx = quantize_mx_op(A=x,
                            mx_specs=self.my_specs, 
                            elem_format=self.elem_format, 
                            block_size=self.block_size, 
                            axes=[-1],
                            round="nearest", 
                            expand_and_reshape=False)
        return qx

if __name__ == "__main__":
    x = torch.linspace(-4.9, 128, 4096).reshape(64, 64)
    print(x)
    np.savetxt("x.txt", x.numpy(), fmt="%.8f")
    mx_int8 = MXQuantizer(elem_format="int8", block_size=32, scale_bits=8)
    qx_int8 = mx_int8.quantize(x)
    print(qx_int8.unique())
    np.savetxt("qx_int8_32.txt", qx_int8.numpy(), fmt="%.8f")