# https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# class FocalLoss(nn.Module): # orig
#     'Focal Loss - https://arxiv.org/abs/1708.02002'

#     def __init__(self, alpha=0.25, gamma=2):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma

#     def forward(self, pred_logits, target):
#         pred = pred_logits.sigmoid()
#         ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
#         alpha = target * self.alpha + (1. - target) * (1. - self.alpha)
#         pt = torch.where(target == 1,  pred, 1 - pred)
#         return alpha * (1. - pt) ** self.gamma * ce


class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


# class FocalLoss(nn.Module):
#     'Focal Loss - https://arxiv.org/abs/1708.02002'

#     def __init__(self, alpha=0.25, gamma=2):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma

#     def forward(self, pred_log_prob, target):
#         # pred = pred_logits.sigmoid()
#         pred = pred_log_prob.exp()
#         ce = F.binary_cross_entropy(pred, target, reduction='none')
#         alpha = target * self.alpha + (1. - target) * (1. - self.alpha)
#         pt = torch.where(target == 1,  pred, 1 - pred)
#         return alpha * (1. - pt) ** self.gamma * ce

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)

#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)

#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
#         else: return loss.sum()


class FocalLossWithOutOneHot(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLossWithOutOneHot, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
        logit = logit.clamp(self.eps, 1. - self.eps)
        logit_ls = torch.log(logit)
        import pdb; pdb.set_trace()
        loss = F.nll_loss(logit_ls, target, reduction="none")
        view = target.size() + (1,)
        index = target.view(*view)
        loss = loss * (1 - logit.gather(1, index).squeeze(1)) ** self.gamma # focal loss

        return loss.sum()
