import os, sys

this_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(this_dir))

sys.path.append(root_dir)
import path_defs
