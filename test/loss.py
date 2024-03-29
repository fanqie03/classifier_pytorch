import torch
from torch import nn

weight = list(range(1,6))
weight = torch.Tensor(weight)
loss = nn.CrossEntropyLoss(weight=weight)
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()