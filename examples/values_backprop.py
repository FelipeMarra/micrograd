#%%
import sys
import os
sys.path.append(os.path.pardir)

from micrograd.engine import *
from micrograd.viz import *

#%%
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
d = a * b; d._label = 'd'
e = d + c; e._label = 'e'
f = Value(-2.0, label='f')
L = e * f; L._label = 'L'
L

# %%
draw_dot(L)

# %%
# Now calculating the gradients from L, 
# wich is our loss function, all the way back
# to the weights a and b

# BACKPROPAGATION:
# First dL(e, f)/dL = 1
def dl_dl():
    h = 0.001
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    d = a * b; d._label = 'd'
    e = d + c; e._label = 'e'
    f = Value(-2.0, label='f')
    L = e * f; L._label = 'L'
    # Basically doing h/h
    L1 = L.data + h
    return (L1 - L.data) / h # Same as (L + h - L) / h
dl_dl()
L.grad = 1
draw_dot(L)

# %%
# dL/de = f and dL/df = e
# In this case, since we are not using the chain rule yet
# the * operator, in respect to the gradients, will just
# trade the operands values
# So that in L = e * f, e.grad = f.data & f.grad = e.data 
def dl_de():
    h = 0.001
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    d = a * b; d._label = 'd'
    e = d + c; e._label = 'e'
    f = Value(-2.0, label='f')
    L = e * f; L._label = 'L'

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    d = a * b; d._label = 'd'
    e = d + c + Value(h); e._label = 'e'
    f = Value(-2.0, label='f')
    L1 = e * f;
    return (L1.data - L.data)/h
dl_de() # the value of f
e.grad = f.data
f.grad = e.data
draw_dot(L)

# %%
# dL/dd = dL/de * de/dd (explanation about the chain 
# rule are in the the_derivative.py)
# dL/dd = e.grad * 1 (= f.data)
# dL/dc = dL/de * de/dc 
# dL/dc = e.grad * 1 (= f.data)
# That leads to the hypothesis that in 
# respect to the gradients, the + operator will
# just rout the gradient of the child node

def dl_dd():
    h = 0.001
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    d = a * b; d._label = 'd'
    e = d + c; e._label = 'e'
    f = Value(-2.0, label='f')
    L = e * f; L._label = 'L'

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    d = a * b + Value(h); d._label = 'd'
    e = d + c; e._label = 'e'
    f = Value(-2.0, label='f')
    L1 = e * f;
    return (L1.data - L.data)/h
dl_de() # the value of e.grad = f.data = -2
c.grad = e.grad
d.grad = e.grad
draw_dot(L)

# %%
# For the last two weights a & b, the use of
# the chain rule will lead to
# dL/da = dL/dd * dL/da = d.grad * b
# dL/db = dL/dd * dL/db = d.grad * a
# So the * operator still trades the Values datas, 
# but also multiply them by the child's grad 
a.grad = b.data * d.grad
b.grad = a.data * d.grad
draw_dot(L)