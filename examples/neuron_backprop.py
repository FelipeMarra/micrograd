#%%
import sys
import os
sys.path.append(os.path.pardir)

from micrograd.engine import *
from micrograd.viz import *

#%%
# BACKPROPAGATION FOR A NEURON:
# inputs x1, x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
#wights w1, w2
w1 = Value(-3, label='w1')
w2 = Value(1.0, label='w2')
#bias 
b = Value(6.8813735870195432, label='b')
#x1w1 + x2w2 + b
x1w1 = x1 * w1; x1w1._label='x1w1'
x2w2 = x2 * w2; x2w2._label='x2w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2._label='x1w1 + x2w2'
n = x1w1x2w2 + b; n._label='n'
o = n.tanh(); o._label = 'out'
draw_dot(o)

# %%
# Manual Backprop
o.grad = 1.0
o._backward()
n._backward()
b._backward()
x1w1x2w2._backward()
x2w2._backward()
x1w1._backward()
draw_dot(o)

# %%
# Auto Backprop
o.backward()
draw_dot(o)

#%%
2 + Value(2)
# %%
