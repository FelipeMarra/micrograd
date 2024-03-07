#%%
import sys
import os
sys.path.append(os.path.pardir)

from micrograd.engine import *
from micrograd.nn import *
from micrograd.viz import *

#%%
# A Neuron
x = [2.0, 3.0]
n = Neuron(2)
n(x)

# %%
# A Layer of Neurons
l = Layer(2, 3)
l(x)

# %%
# A MLP
m = MLP(3, [4,4,1])
m(x)
draw_dot(m(x))
# %%
