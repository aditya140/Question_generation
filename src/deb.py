import torch
a=torch.rand(1)[0]<torch.tensor(0.8)
if a:
    print(1)
print(a)

