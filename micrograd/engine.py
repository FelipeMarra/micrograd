#%%
import math

#%% 
class Value:
    """
        data = the value contained in this obj

        grad = The derivative of this Value
        with respect to the loss function in 
        backprop. Initialized with 0

        _prev = a tuple of Values that created
        this one through a supported operation,
        e.g., in c = a + b the prevs of c are 
        a and b

        _op = the operation used to create this
        value, e.g, in c = a + b the op of c
        is +

        _label = user defined "variable name", e.g, 
        c = a + b; c._label = "c". Used in the __str__ 
        function
    """

    def __init__(self, data:float, prev:tuple=(), op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(prev)
        self._op = op
        self._label = label

    def __str__(self) -> str:
        return f"Value {self._label}: {self.data}"

    def __repr__(self) -> str:
        return f"Value({self.data}, {self._op}, {self._label})"

#################       Operators        #############################
    def __add__(self, other):
        """
            When the + operator is used, return
            a Value obj containing the data in this
            value + the data from the other value

            The logic will be the same for every other 
            operator
        """
        other = other if isinstance(other, Value) else Value(other)
        data = self.data + other.data
        prev = (self, other)
        out = Value(data, prev, '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out 

    def __radd__(self, other):
        """
            This function is called in cases like 2 + a,
            where 2 doesn't know how to be added to the Value a.
        """
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        data = self.data * other.data
        prev = (self, other)
        out = Value(data, prev, '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward

        return out

    def __rmul__(self, other):
        """
            This function is called in cases like 2 * a,
            where 2 doesn't know how to be multiplied by the Value a.
            Here we are inverting the operation to a * 2, 
            so that a__mul__(2) will be executed instead.
        """
        return self * other

    def __truediv__(self, other):
        """
            Division of a by b (a/b), where a and b are Values.
            Obviously to divide is the same as to multiply by the reciprocal.
        """
        return self * (other**-1)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supports ^(int or float)"
        data = self.data**other
        out = Value(data, (self, ), f"^{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        data = math.exp(self.data)
        out = Value(data, (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

################# Activation Functions #############################
    def tanh(self):
        x = self.data
        # e^(2x) -1 / e^(2x) +1
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

#################       Backprop          #############################
    def backward(self):
        topo = []
        visited = set()

        # Topological Sorting: wikipedia.org/wiki/Topological_sorting
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for prev in v._prev:  
                    build_topo(prev)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

        self.grad = 1
