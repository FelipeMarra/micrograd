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
print(l(x))
l.parameters()

# %%
# A MLP
mlp = MLP(3, [4,4,1])
##print(m(x))
#m.parameters()
#draw_dot(m(x))

# %%
# Dataset
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0] # desired targets

#%% 
# Predictions
def predict(mlp, xs):
    return [mlp(x) for x in xs]

y_hats = predict(mlp, xs)

#%%
# Loss
def loss(y_hats, ys):
    return sum((y_hat - y)**2 for y_hat,y in zip(y_hats, ys))

loss_value = loss(y_hats, ys)
loss_value

#%%
#Backward pass
loss_value.backward()

#%%
print('First neuron grad: ', mlp.layers[0].neurons[0].w[0].grad)
print('First neuron data: ', mlp.layers[0].neurons[0].w[0].data)

# %%
# Gradient Descent

# Lets say we have a loss of 5 and a neuron n w/ a
# negative grad, like -0.3, w/ its data equals 0.85.
# Since the grad (derivative of the loss in relation 
# to the neuron data) is negative, we know that
# increase the data, like to 0.9, will decrese the loss 
# like to 4.95 (loss == prediction error). 

# Adding n.grad * STEP to data would decrease the data,
# decreasing its negative influence in the loss
# (making the error go up). That's why gradient descent 
# uses n.grad * -STEP
STEP = 1e-1
EPOCHS = 25

for e in range(EPOCHS):
    # forward
    y_hats = predict(mlp, xs)
    loss_value = loss(y_hats, ys)

    # zero grad
    for p in mlp.parameters():
        p.grad = 0

    # backward
    loss_value.backward()

    # update
    for p in mlp.parameters():
        p.data += p.grad * -STEP

    print(f"EPOCH {e}, LOSS {loss_value.data}")

#%%
y_hats = predict(mlp, xs)
y_hats
# %%
