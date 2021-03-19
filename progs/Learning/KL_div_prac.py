import torch
import torch.nn as nn
import torch.nn.functional as F

kldiv_criterion = torch.nn.KLDivLoss(reduction='none')

inputs = torch.Tensor([0.36, 0.48, 0.16])
targets = torch.Tensor([0.333, 0.333, 0.333])

print(inputs.sum(), targets.sum())

kldiv_loss = kldiv_criterion(targets.log(), inputs)
print(kldiv_loss.sum())

kldiv_loss = kldiv_criterion(inputs.log(), targets)
print(kldiv_loss.sum())




## Simulation in NN situation
kldiv_criterion = nn.KLDivLoss(reduction='none')

def print_kldiv_loss(probs, targets):
  # probs = torch.Tensor([0.9, 0.1, 0., 0.])
  log_probs = torch.log(probs + 1e-5)
  # print(log_probs.exp().sum()) # 1.000

  # targets = torch.Tensor([0, 0., 0., 1.])

  # logits = torch.Tensor([10, 0, -10]) # NN outputs
  # log_probs = F.log_softmax(logits, dim=-1)


  kldiv_loss = kldiv_criterion(log_probs, targets).sum()
  print(kldiv_loss)


probs = torch.Tensor(4)
targets = torch.Tensor([0, 0., 0., 1.])
print(kldiv_criterion(probs.log(), targets))

'''
# this is the same example in wiki
P = torch.Tensor([0.36, 0.48, 0.16]) # log-probabilities
Q = torch.Tensor([0.333, 0.333, 0.333]) # probabilities

print((P * (P / Q).log()).sum())

import torch.nn as nn
kldiv = nn.KLDivLoss(reduction="sum")
print(kldiv(Q.log(), P))
'''
