import os
import torch
import torch.nn.functional as F

words = open(os.path.join(os.path.dirname(__file__), 'names.txt')).read().splitlines()

chars = sorted(set(''.join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

xs = []
ys = []


for word in words:
    word = "." + word + "."
    for i in range(len(word) - 1):
        ix1 = stoi[word[i]]
        ix2 = stoi[word[i+1]] 
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

xenc = F.one_hot(xs, num_classes=27).float()

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

for k in range(200):
    # forward pass
    logits = xenc @ W # (N, 27) - raw scores, interpret as log-counts
    counts = logits.exp() # exponentiate -> all positive, like counts
    probs = counts / counts.sum(1, keepdim=True) # normalize each row -> probabilities

    num = xs.nelement()
    loss = -probs[torch.arange(num), ys].log().mean()

    # backward pass
    W.grad = None # reset gradients to zero
    loss.backward() # compute dloss/dW

    # update
    W.data += -50 * W.grad # step downhill; 50 is the learning rate
    print(loss.item())