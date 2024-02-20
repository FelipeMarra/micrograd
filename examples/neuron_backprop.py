#%%
import sys
import os
sys.path.append(os.path.pardir)

from micrograd.engine import *
from micrograd.viz import *
#%%
# BACKPROPAGATION FOR A NEURON:
# inputs x1, x2