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
        dot.node(name=uid, label="{ %s | data %.4f}" % (n._label, n.data), shape='record')
        
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
