import os
import torch
import torch.nn.functional as F

words = open(os.path.join(os.path.dirname(__file__), 'names.txt')).read().splitlines()

chars = sorted(set(''.join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

block_size = 3
X, Y = [], []
for word in words:
    context = [0] * block_size          # start padded with '.'
    for ch in word + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]    # slide the window
X = torch.tensor(X)
Y = torch.tensor(Y)

C = torch.randn((27,2)) # 27 chars, each embedded into a 2-D  vector

W1 = torch.randn((6, 100))    # 6 = 3 chars × 2 dims, 100 hidden units
b1 = torch.randn(100)

W2 = torch.randn((100, 27))
b2 = torch.randn(27)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True


for k in range(30000):
    ix = torch.randint(0, X.shape[0], (32,)) # 32 random example indices
    # forward
    emb = C[X[ix]] # index X by ix
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y[ix])# index Y by ix
    # backward
    for p in parameters:
        p.grad = None
    loss.backward()
    # update
    lr = 0.1 if k < 20000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad
    print(loss.item())

emb = C[X]
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y)
print(loss.item())

