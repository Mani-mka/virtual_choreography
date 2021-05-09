import torch

x = torch.randn(2, 3).type(dtype=torch.float64)
y = torch.cat((x, x, x), 0)

print(y)