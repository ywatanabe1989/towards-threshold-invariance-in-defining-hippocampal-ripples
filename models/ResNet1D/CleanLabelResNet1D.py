#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator

import sys; sys.path.append('.')
from models.ResNet1D.ResNet1D import ResNet1D
from modules.ranger2020 import Ranger

import utils.general as ug


## Fix seeds
ug.fix_seeds(os=None, random=None, np=np, torch=torch, tf=None, seed=42)

class CleanLabelResNet1D(BaseEstimator):
    '''Wraps ResNet1D written in PyTorch for okada's ripple dataset within an sklearn template
    by defining the following methods: self.fit(), self.predict(), and self.predict_proba().
    This template enables ResNet1D to flexibly be used in sklearn's architecture; the model can be
    passed to functions like cross_val_predict. The cleanlab library requires all models adhere to
    this basic sklearn template to use their providing functions easily.
    '''
    def __init__(self, resnet1d_conf, dl_conf):
        self.resnet1d_conf = resnet1d_conf
        self.dl_conf = dl_conf
        
        self.resnet1d = ResNet1D(self.resnet1d_conf).to(self.resnet1d_conf['device'])
        self.softmax = nn.Softmax(dim=-1)
        self.xentropy_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.loss_tra = []
        self.log_interval = 50
        
        
    def fit(self, X, T, sample_weight=None, max_epochs=30):
        '''This method adheres to sklearn's "fit(X, y)" format for compatibility with scikit-learn.
        All inputs should be numpy arrays (not PyTorch Tensors).
        train_idx is not X, but instead a list of indices for X (and y if train_labels is None).
        '''
        ds_tra = torch.utils.data.TensorDataset(torch.FloatTensor(X),
                                                torch.LongTensor(T))
        
        dl_tra = torch.utils.data.DataLoader(dataset=ds_tra,
                                             **self.dl_conf,
                                             shuffle=True,
                                  )    
        
        optimizer = Ranger(self.resnet1d.parameters(), lr=self.resnet1d_conf['lr'])

        # Train for self.epochs epochs
        self.resnet1d.train()        
        for i_epoch, epoch in enumerate(range(1, max_epochs + 1)):            
            for i_batch, batch in enumerate(dl_tra):
                Xb, Tb = batch
                Xb, Tb = Xb.to(self.resnet1d_conf['device']), Tb.to(self.resnet1d_conf['device'])
                    
                # Xb, Tb = Variable(Xb), Variable(Tb)
                optimizer.zero_grad()
                y = self.resnet1d(Xb)
                loss = self.xentropy_criterion(y, Tb.long())
                self.loss_tra.append(loss.item())
                loss.backward()
                optimizer.step()
                
                # if self.log_interval is not None and i_batch % self.log_interval == 0:
                if i_batch % self.log_interval == 0:
                    print('Train Epoch: {e} [{n_seen}/{n_tra} ({seen_perc:.0f})]\tLoss: {l:.6f}'\
                          .format(e=epoch,
                                  n_seen=i_batch * len(Xb),
                                  n_tra=len(X),
                                  seen_perc=100. * i_batch / len(dl_tra),
                                  l=loss.item(),
                                  )
                          )

    
    def predict(self, X):
        pass
    # def predict(self, idx = None, loader = None):
    #     # get the index of the max probability
    #     probs = self.predict_proba(idx, loader)
    #     return probs.argmax(axis=1)
    
    def predict_proba(self, X):
        ## Create Dataloader
        ds_tes = torch.utils.data.TensorDataset(torch.FloatTensor(X))
        
        dl_tes = torch.utils.data.DataLoader(dataset=ds_tes,
                                             **self.dl_conf, shuffle=False,

                                  )    

        self.resnet1d.eval()

        # Run forward pass on model to compute outputs
        outs = []
        for i_batch, batch in enumerate(dl_tes):
            Xb = batch[0]
            Xb = Xb.to(self.resnet1d_conf['device'])

            y = self.resnet1d(Xb)
            outs.append(y.detach().cpu())

        outs = torch.cat(outs, dim=0)
        pred = self.softmax(outs)

        return pred
    
    def score(self, X, y, sample_weight=None):
        pass    



if __name__ == '__main__':
    from utils.general import load_yaml_as_dict
    from sklearn.model_selection import train_test_split


    ## Fix seeds
    ug.fix_seeds()    
    
    ## Data
    labels= ['nonRipple', 'Ripple']
    n_classes = len(labels)
    n_all = 256
    bs, n_chs, seq_len = 16, 1, 400
    X_all = torch.rand(n_all, n_chs, seq_len)
    T_all = torch.randint(0,2,(n_all,))
    X_tra, X_tes, T_tra, T_tes = train_test_split(X_all, T_all, random_state=42)

    ## Model
    model_conf = load_yaml_as_dict('./models/ResNet1D/CleanLabelResNet1D.yaml')
    model_conf['labels'] = labels
    model_conf['n_chs'] = 1
    model_conf['seq_len'] = seq_len
    model_conf['lr'] = 1e-3
    model_conf['device'] = 'cuda'

    dl_conf = {'batch_size': bs,
               'num_workers': 10,
               'drop_last': True,
               }
    model = CleanLabelResNet1D(model_conf, dl_conf)

    ## Training
    model.fit(X_tra, T_tra)

    ## Evaluation
    pred_class_tra = model.predict_proba(X_tra).argmax(dim=-1)
    acc_tra = (pred_class_tra == T_tra).float().mean()
    print(acc_tra) # 1.

    pred_class_tes = model.predict_proba(X_tes).argmax(dim=-1)
    acc_tes = (pred_class_tes == T_tes).float().mean()
    print(acc_tes) # 0.7188

    ## EOF
