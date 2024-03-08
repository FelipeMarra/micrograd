from micrograd.engine import *
import random

class Neuron():
    """
        in_size: number of inputs
    """
    def __init__(self, in_size:int) -> None:
        self.w = [Value(random.uniform(-1,1)) for _ in range(in_size)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x:list[Value]) -> Value:
        """
            Allows one to call an object as 
            a function, e.g., n = Neuron()
            n() -> calls the __call__ method

            Here it will be used to calculate 
            a neuron's linear combination, that
            is, w * x + b
        """
        linear = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return linear.tanh()

    def parameters(self):
        return self.w + [self.b]

class Layer():
    def __init__(self, in_size:int, layer_size:int) -> None:
        self.neurons = [Neuron(in_size) for _ in range(layer_size)]

    def __call__(self, x) -> list[Value]:
        activations = []

        for n in self.neurons:
            activations.append(n(x))

        return activations[0] if len(activations) == 1 else activations

    def parameters(self) -> list[Value]:
        # params = []
        # for neuron in self.neurons:
        #     params.extend(neuron.parameters())
        # return params
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP():
    def __init__(self, in_size:int, layers_sizes:list[int]) -> None:
        sizes = [in_size] + layers_sizes
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(layers_sizes))]

    def __call__(self, x) -> list[Value]:
        for l in self.layers:
            x = l(x)

        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]