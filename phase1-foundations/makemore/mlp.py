import os
import torch
import torch.nn.functional as F
import random


def build_dataset(words):
    X, Y = [], []
    for word in words:
        context = [0] * block_size          # start padded with '.'
        for ch in word + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]    # slide the window
    return torch.tensor(X), torch.tensor(Y)

words = open(os.path.join(os.path.dirname(__file__), 'names.txt')).read().splitlines()

chars = sorted(set(''.join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
block_size = 3

random.seed(42)
random.shuffle(words)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1]) # 80% train
Xdev, Ydev = build_dataset(words[n1:n2]) # 10% dev
Xte, Yte = build_dataset(words[n2:]) # 10% test

emb_dim = 10
n_hidden = 200
C = torch.randn((27,emb_dim))
W1 = torch.randn((block_size * emb_dim, n_hidden))
b1 = torch.randn(n_hidden)
W2 = torch.randn((n_hidden, 27))
b2 = torch.randn(27)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True


for k in range(100000):
    ix = torch.randint(0, Xtr.shape[0], (64,)) # 64 random example indices
    # forward
    emb = C[Xtr[ix]] # index X by ix
    h = torch.tanh(emb.view(-1, block_size * emb_dim) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])# index Y by ix
    # backward
    for p in parameters:
        p.grad = None
    loss.backward()
    # update
    lr = 0.1 if k < 70000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad
    print(loss.item())

#sampling
g = torch.Generator().manual_seed(2147483647)
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))


emb = C[Xtr]
h = torch.tanh(emb.view(-1, block_size * emb_dim) @ W1 + b1)
loss = F.cross_entropy(h @ W2 + b2, Ytr)
print('train', loss.item())

emb = C[Xdev]
h = torch.tanh(emb.view(-1, block_size * emb_dim) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print('dev', loss.item())

emb = C[Xte]
h = torch.tanh(emb.view(-1, block_size * emb_dim) @ W1 + b1)
loss = F.cross_entropy(h @ W2 + b2, Yte)
print('test', loss.item())


