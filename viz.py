#%%
from micrograd.engine import *
from graphviz import Digraph

#%%
def trace(root) -> tuple[tuple[Value], tuple[Value, Value]]:
    """
        Builds a set of all nodes and edges in a graph
    """
    nodes, edges = set(), set()

    def build(v: Value):
        if v not in nodes:
            nodes.add(v)
            for prev in v._prev:
                edges.add((prev, v))
                build(prev)

    build(root)
    return nodes, edges

def draw_dot(root: Value):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph create a rectangular node for it
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f}" % (n._label, n.data, n.grad), shape='record')
        
        if n._op:
            # If this value is the result of an operation, create an op node for it
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connet n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

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
# Now calculating the gradients from L, wich is our loss function, all the way back
# to the wights a and b
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
# dL/dd = dL/de * de/dd (explanation about the chain rule are in the the_derivative.py)
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
# For the last two weights a & b, the use of the chain rule will lead to
# dL/da = dL/dd * dL/da = d.grad * b
# dL/db = dL/dd * dL/db = d.grad * a
# So the * operator still trades the Values datas, but also multiply them
# by the child's grad 
a.grad = b.data * d.grad
b.grad = a.data * d.grad
draw_dot(L)
