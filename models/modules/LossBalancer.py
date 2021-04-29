#!/usr/bin/env python
import torch
import numpy as np
import torch.nn as nn
# import pytorch_lightning as pl


class LossBalancer(nn.Module):
    """Balance the cost originated from an imbalanced dataset with regard to the sample sizes in an online manner.
    """
    def __init__(self, n_classes_int, bs):
        super().__init__()
        self.register_buffer('n_classes', torch.IntTensor([n_classes_int]))
        # self.n_classes = torch.IntTensor([n_classes_int], device=self.device)

        self.register_buffer('cum_n_samp_per_cls', torch.zeros(n_classes_int,
                                              dtype=torch.int64))
        # self.cum_n_samp_per_cls = torch.zeros(n_classes_int,
        #                                       dtype=torch.int64,
        #                                       device=self.device)

        self.register_buffer('weights_norm', torch.ones(bs) / self.n_classes)
                             
        # self.weights_norm = torch.ones(bs, device=self.device) / self.n_classes
        
    def __call__(self, loss, Tb, i_current_epoch, train=True):
        self.loss_device, self.loss_dtype = loss.device, loss.dtype

        self.weights_norm = self.weights_norm.to(self.loss_device).to(self.loss_dtype)
        
        if (i_current_epoch == 0) & (train is True):
            Tb = Tb.to(self.loss_device)
            self._update_sample_counter(Tb)
            self._update_weights(loss, Tb)

        loss *= self.weights_norm # Balance the Loss
        return loss

    def _update_sample_counter(self, Tb):
        Tb_onehot = self._to_onehot(Tb, self.n_classes)
        self.cum_n_samp_per_cls += Tb_onehot.sum(dim=0)

    def _to_onehot(self, label, k):
        return torch.eye(int(k))[label].int()
            
    def _update_weights(self, loss, Tb):
        weights = torch.zeros_like(Tb, dtype=self.loss_dtype).to(self.loss_device) #.to(device)
        probs_arr = (1. * self.cum_n_samp_per_cls / self.cum_n_samp_per_cls.sum())\
            .to(self.loss_dtype).to(self.loss_device)
        non_zero_mask = (probs_arr > 0)
        recip_probs_arr = torch.zeros_like(probs_arr, dtype=self.loss_dtype).to(self.loss_device)
        recip_probs_arr[non_zero_mask] = probs_arr[non_zero_mask] ** (-1)
        for i in range(self.n_classes):
            mask = (Tb == i)
            weights[mask] += recip_probs_arr[i] # weights[mask] torch.int64
        self.weights_norm = (weights / weights.mean())



# class LossBalancer(nn.Module):
#     """Balance the cost originated from an imbalanced dataset with regard to the sample sizes in an online manner.
#     """
#     def __init__(self, n_classes_int, bs):
#         super().__init__()        
#         self.n_classes = torch.IntTensor([n_classes_int], device=self.device)
        
#         self.cum_n_samp_per_cls = torch.zeros(n_classes_int,
#                                               dtype=torch.int64,
#                                               device=self.device)
        
#         self.weights_norm = torch.ones(bs, device=self.device) / self.n_classes
        
#     def __call__(self, loss, Tb, i_current_epoch, train=True):
#         self.loss_device, self.loss_dtype = loss.device, loss.dtype

#         self.weights_norm = self.weights_norm.to(self.loss_device).to(self.loss_dtype)
        
#         if (i_current_epoch == 0) & (train is True):
#             Tb = Tb.to(self.loss_device)
#             self._update_sample_counter(Tb)
#             self._update_weights(loss, Tb)

#         loss *= self.weights_norm # Balance the Loss
#         return loss

#     def _update_sample_counter(self, Tb):
#         Tb_onehot = self._to_onehot(Tb, self.n_classes)
#         self.cum_n_samp_per_cls += Tb_onehot.sum(dim=0)

#     def _to_onehot(self, label, k):
#         return torch.eye(int(k))[label].int()
            
#     def _update_weights(self, loss, Tb):
#         weights = torch.zeros_like(Tb, dtype=self.loss_dtype).to(self.loss_device) #.to(device)
#         probs_arr = (1. * self.cum_n_samp_per_cls / self.cum_n_samp_per_cls.sum())\
#             .to(self.loss_dtype).to(self.loss_device)
#         non_zero_mask = (probs_arr > 0)
#         recip_probs_arr = torch.zeros_like(probs_arr, dtype=self.loss_dtype).to(self.loss_device)
#         recip_probs_arr[non_zero_mask] = probs_arr[non_zero_mask] ** (-1)
#         for i in range(self.n_classes):
#             mask = (Tb == i)
#             weights[mask] += recip_probs_arr[i] # weights[mask] torch.int64
#         self.weights_norm = (weights / weights.mean())




        

# class LossBalancer(pl.LightningModule):
#     # https://pytorch-lightning.readthedocs.io/en/stable/performance.html
#     """Balance the cost originated from an imbalanced dataset with regard to the sample sizes in an online manner.
#     """
#     def __init__(self, n_classes_int, bs):
#         super(LossBalancer, self).__init__()        
#         self.n_classes = torch.IntTensor([n_classes_int], device=self.device)
        
#         self.cum_n_samp_per_cls = torch.zeros(n_classes_int,
#                                               dtype=torch.int64,
#                                               device=self.device)
        
#         self.weights_norm = torch.ones(bs, device=self.device) / self.n_classes
        
#     def __call__(self, loss, Tb, i_current_epoch, train=True):
#         self.loss_device, self.loss_dtype = loss.device, loss.dtype

#         self.weights_norm = self.weights_norm.to(self.loss_device).to(self.loss_dtype)
        
#         if (i_current_epoch == 0) & (train is True):
#             Tb = Tb.to(self.loss_device)
#             self._update_sample_counter(Tb)
#             self._update_weights(loss, Tb)

#         loss *= self.weights_norm # Balance the Loss
#         return loss

#     def _update_sample_counter(self, Tb):
#         Tb_onehot = self._to_onehot(Tb, self.n_classes)
#         self.cum_n_samp_per_cls += Tb_onehot.sum(dim=0)

#     def _to_onehot(self, label, k):
#         return torch.eye(int(k))[label].int()
            
#     def _update_weights(self, loss, Tb):
#         weights = torch.zeros_like(Tb, dtype=self.loss_dtype).to(self.loss_device) #.to(device)
#         probs_arr = (1. * self.cum_n_samp_per_cls / self.cum_n_samp_per_cls.sum())\
#             .to(self.loss_dtype).to(self.loss_device)
#         non_zero_mask = (probs_arr > 0)
#         recip_probs_arr = torch.zeros_like(probs_arr, dtype=self.loss_dtype).to(self.loss_device)
#         recip_probs_arr[non_zero_mask] = probs_arr[non_zero_mask] ** (-1)
#         for i in range(self.n_classes):
#             mask = (Tb == i)
#             weights[mask] += recip_probs_arr[i] # weights[mask] torch.int64
#         self.weights_norm = (weights / weights.mean())
