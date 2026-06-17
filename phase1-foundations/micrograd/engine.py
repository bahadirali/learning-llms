import math

class Value:
    def __init__(self, x, prev = (), op = ''):
        self.x = x
        self.prev = set(prev)
        self.op = op
        self.grad = 0.0
        self._backward = lambda: None
    def __repr__(self):
        return f"{self.__class__.__name__}({self.x})"
    def __add__(self, other):
        out = Value(self.x + other.x, (self, other), '+')
        def _backward():
           self.grad += 1.0 * out.grad
           other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    def __mul__(self, other):
        out = Value(self.x * other.x, (self, other), '*')
        def _backward():
            self.grad += other.x * out.grad
            other.grad += self.x * out.grad
        out._backward = _backward
        return out
    def tanh(self):
        out = Value(math.tanh(self.x), (self,), 'tanh')
        def _backward():
           self.grad += (1.0 - out.x**2) * out.grad
        out._backward = _backward
        return out
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v in visited: return
            visited.add(v)
            for child in v.prev:
                build_topo(child)
            topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


    
    
a = Value(0.8)
b = a.tanh()
b.backward()
# Expected: b.x ≈ 0.6640, a.grad ≈ 1 - 0.6640**2 ≈ 0.5591
print(b.x, a.grad)



