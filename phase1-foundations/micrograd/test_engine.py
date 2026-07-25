import torch
from engine import Value

a = Value(-4.0)
ta = torch.tensor([-4.0], dtype=torch.double, requires_grad=True)
b = a * 2 + a**3 - a/4
tb = ta * 2 + ta**3 - ta/4
b.backward()
tb.backward()
assert abs(a.grad - ta.grad.item()) < 1e-9
