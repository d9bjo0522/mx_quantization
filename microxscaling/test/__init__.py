# Initialize test package
import sys
import os

# Get the current directory of the __init__.py file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory of 'mx' to the system path
sys.path.insert(0, os.path.join(current_dir, '../../'))