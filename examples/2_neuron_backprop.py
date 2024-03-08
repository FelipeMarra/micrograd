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
# inputs x1, x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
#wights w1, w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
#bias 
b = Value(6.8813735870195432, label='b')
#x1w1 + x2w2 + b
x1w1 = x1 * w1; x1w1._label='x1w1'
x2w2 = x2 * w2; x2w2._label='x2w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2._label='x1w1 + x2w2'
n = x1w1x2w2 + b; n._label='n'
## tanh by hand to test operations ##
e = (2*n).exp()
o = (e - 1) / (e + 1); o._label = 'o'
o.backward()
draw_dot(o)

# %%
# Torch version
import torch

# inputs x1, x2
x1 = torch.tensor([2.0], dtype=torch.double, requires_grad=True)
x2 = torch.tensor([0.0], dtype=torch.double, requires_grad=True)
#wights w1, w2
w1 = torch.tensor([-3.0], dtype=torch.double, requires_grad=True)
w2 = torch.tensor([1.0], dtype=torch.double, requires_grad=True)
#bias 
b = torch.tensor([6.8813735870195432], dtype=torch.double, requires_grad=True)
#x1w1 + x2w2 + b
n = x1*w1 + x2*w2 + b
o = n.tanh()

print('o:',o.item())

o.backward()

print('x1:',x1.grad.item())
print('w1:',w1.grad.item())

print('x2:',x2.grad.item())
print('w2:',w2.grad.item())

# %%
