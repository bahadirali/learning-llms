import os
import torch

words = open(os.path.join(os.path.dirname(__file__), 'names.txt')).read().splitlines()

chars = sorted(set(''.join(words)))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

# Create a 27x27 tensor of zeros to hold bigram counts
N = torch.zeros((27, 27), dtype=torch.int32)


for word in words:
    word = "." + word + "."
    for i in range(len(word) - 1):
        ix1 = stoi[word[i]]
        ix2 = stoi[word[i+1]] 
        N[ix1, ix2] += 1



P = N.float()
P = P / P.sum(1, keepdim = True)

log_likelihood = 0.0
n = 0
for word in words:
    word = "." + word + "."
    for i in range(len(word) - 1):
        ix1 = stoi[word[i]]
        ix2 = stoi[word[i+1]]
        log_likelihood += torch.log(P[ix1, ix2])
        n += 1

nll = -log_likelihood / n
print(f"{nll.item():.4f}")

# g = torch.Generator().manual_seed(2147483647)
# generated_names = []
# for _ in range(10):
#     s: str = ""
#     ch = "."
#     while True:
#         next_i = torch.multinomial(P[stoi[ch]], num_samples=1, replacement=True, generator=g)
#         ch = itos[next_i[0].item()]
#         if ch ==".":
#             break
#         s += ch
#     generated_names.append(s)

# for name in generated_names:
#     print(name)