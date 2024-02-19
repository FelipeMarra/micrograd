#%% 
class Value:
    """
        data = the value contained in this obj

        _prev = a tuple of Values that created
        this one through a supported operation,
        e.g., in c = a + b the prevs of c are 
        a and b

        _op = the operation used to create this
        value, e.g, in c = a + b the op of c
        is +

        _label = string for better identification
    """

    def __init__(self, data:float, prev:tuple=(), op='', label=''):
        self.data = data
        self._prev = set(prev)
        self._op = op
        self._label = label

    def __repr__(self) -> str:
        return f"Value {self._label}: {self.data}"

    def __add__(self, other):
        """
            When the + operator is used, return
            a Value obj containing the data in this
            value + the data from the other value

            The logic will be the same for every other 
            operator
        """
        data = self.data + other.data
        prev = (self, other)
        return Value(data, prev, '+')

    def __mul__(self, other):
        data = self.data * other.data
        prev = (self, other)
        return Value(data, prev, '*')
# %%
