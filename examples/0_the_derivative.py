# Derivatives mesure how a function at a point x 
# reponds to a small positive increase h. 
# With what sensitivity it responds? It goes up
# or dow? And by how much? All of this is 
# represented by the slop of that reponse at that 
# point:
# lim f(x) h->0 = (f(x + h) - f(x)) / h

#%%
import numpy as np
import matplotlib.pyplot as plt

#%% 
# Define some function f
def f(x):
    return 3*x**2 -4*x + 5

#%%
# f of 3 is 20
f(3)

# %%
# Ploting f from -5 to 5 in steps of 0.25
xs = np.arange(-5, 5, 0.25)
ys = f(xs)
plt.plot(xs, ys)

# %%
# If we define a small h, f(x + h) tells us how the
# function responds to that small positive increase
h = 0.00001
x = 3
f(x + h)

# %%
# If we want just the change caused by h, then
f(x + h) - f(x)

# %%
# Normalizing by h we get an aproximation 
# of the derivative or the slop at point
# x (the rise over run, or y2-y1/x2-x1)
(f(x + h) - f(x)) / h 

# %%
# At a point were the values are decreasing, the slop
# will be negative
x = -3
(f(x + h) - f(x)) / h 

# %%
# At what point the funtion doesent repond? Where the
# slop is zero. That will be a mix ou min point
x = 2/3
(f(x + h) - f(x)) / h 
# The value will be of course, only close to 0, since
# it is an aproximation 

# %%
# Now for the derivative for a function with multiple
# inputs
a = 2.0
b = -3.0
c = 10.0

# d is a function of a, b & c
d = a * b + c

d

# %%
# If we are looking to the derivative of d w/ respect
# to a, we expect that the effect will be a decrease
# in the result, because b is negative. Therefore the
# slop is negative
da = (a + h) * b + c
da

# %%
# Taking just the change and normalizing by h, the
# aproximantion of the slop is close to -3, that is,
# it is close to b
print("da - d =", (da - d))
(da - d) / h

# %%
# Now with respect to b, the slop will be a
db = a * (b + h) + c
print("db - d =", (db - d))
(db - d) / h


# %%
# With respect to c the slope goes to 1, since we are
# adding h and dividing by h
dc = a * b + (c + h)
print("dc - d =", (dc - d))
(dc - d) / h

#%%
# The Chain Rule
# https://en.wikipedia.org/wiki/Chain_rule#Intuitive_explanation