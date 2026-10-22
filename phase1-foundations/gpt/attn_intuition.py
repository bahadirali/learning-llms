import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

#weighted-average matrix: lower triangular, rows normalized to sum to 1
tril = torch.tril(torch.ones(T, T))
wei = tril / tril.sum(1, keepdim=True)
#print(wei)

xbow = wei @ x      # (T, T) @ (B, T, C) -> (B, T, C)




head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)      # (B, T, head_size)
q = query(x)      # (B, T, head_size)

wei = q @ k.transpose(-2, -1) * head_size**-0.5     # (B, T, T) - affinities
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))    # causal mask
wei = F.softmax(wei, dim=-1)                        # normalize to weights

v = value(x)    # (B, T, head_size)
out = wei @ v   # (B, T, head_size)
print(out.shape)
print(wei[0])
