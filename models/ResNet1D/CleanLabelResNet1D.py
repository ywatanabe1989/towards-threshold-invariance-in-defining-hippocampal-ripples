#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator

import sys; sys.path.append('.')
from models.ResNet1D.ResNet1D import ResNet1D
from models.modules.LossBalancer import LossBalancer
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
        n_classes = len(self.resnet1d_conf['labels'])
        self.softmax = nn.Softmax(dim=-1)
        self.xentropy_criterion = nn.CrossEntropyLoss(reduction='none')
        self.loss_balancer = LossBalancer(n_classes, dl_conf['batch_size'])
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
                                             drop_last=True,
                                  )    
        
        optimizer = Ranger(self.resnet1d.parameters(), lr=self.resnet1d_conf['lr'])

        # Train for self.epochs epochs
        self.resnet1d.train()        
        for i_epoch in range(1, max_epochs + 1):
            for i_batch, batch in enumerate(dl_tra):
                Xb, Tb = batch
                Xb, Tb = Xb.to(self.resnet1d_conf['device']), Tb.to(self.resnet1d_conf['device'])
                    
                # Xb, Tb = Variable(Xb), Variable(Tb)
                optimizer.zero_grad()
                y = self.resnet1d(Xb)
                loss = self.xentropy_criterion(y, Tb.long())
                loss = self.loss_balancer(loss, Tb, i_epoch, train=True).mean()
                self.loss_tra.append(loss.item())
                loss.backward()
                optimizer.step()
                
                # if self.log_interval is not None and i_batch % self.log_interval == 0:
                if i_batch % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'\
                          .format(i_epoch,
                                  (i_batch+1) * len(Xb),
                                  len(X),
                                  100. * (i_batch+1) * len(Xb) / len(X),
                                  loss.item(),
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
                                             **self.dl_conf,
                                             shuffle=False,
                                             drop_last=False,
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
    from cleanlab.latent_estimation import estimate_confident_joint_and_cv_pred_proba
    from cleanlab.pruning import get_noise_indices
    import numpy as np
    import random    
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.datasets import load_digits

    from utils.general import load_yaml_as_dict
    from utils.Reporter import Reporter


    ## Fix seeds
    ug.fix_seeds(random=random, np=np)

    
    ## Data
    mnist = load_digits(n_class=2)
    X_all, T_all = mnist.data, mnist.target
    labels= ['0', '1']
    n_classes = len(np.unique(T_all))
    bs, n_chs, seq_len = 16, 1, X_all.shape[-1]

    ################################################################################    
    ## Model
    ################################################################################    
    model_conf = load_yaml_as_dict('./models/ResNet1D/CleanLabelResNet1D.yaml')
    model_conf['labels'] = labels
    model_conf['n_chs'] = 1
    model_conf['seq_len'] = seq_len
    model_conf['lr'] = 1e-3
    model_conf['device'] = 'cuda'

    dl_conf = {'batch_size': bs,
               'num_workers': 10,
               }

    ################################################################################
    ## Confident Learning using cleanlab
    ################################################################################    
    model = CleanLabelResNet1D(model_conf, dl_conf)
    # Compute the confident joint and the n x m predicted probabilities matrix (psx),
    # for n examples, m classes. Stop here if all you need is the confident joint.
    confident_joint, psx = estimate_confident_joint_and_cv_pred_proba(
        X=X_all,
        s=T_all,
        clf=model, # default, you can use any classifier
    )

    are_errors = get_noise_indices(T_all,
                                   psx,
                                   inverse_noise_matrix=None,
                                   prune_method='prune_by_noise_rate',
                                   n_jobs=20,
                                   )

    print('\nLabel Errors Indice:\n{}\n'.format(are_errors))

    ## imshow
    indi = np.where(np.array(are_errors) == True)[0]
    for ii in indi:
        Xi, Ti = X_all[ii], T_all[ii]
        plt.imshow(Xi.reshape(8,8))
        plt.title(Ti)
        plt.show()
    

    '''
    ################################################################################    
    ## Main; just trains the model and predicts test data as usual
    ################################################################################
    ## Paramters
    N_FOLDS = 5
    
    reporter = Reporter(sdir='/tmp/')
    skf = StratifiedKFold(n_splits=N_FOLDS)
    # for i_fold in range(N_FOLDS):
    #     X_tra, X_tes, T_tra, T_tes = train_test_split(X_all, T_all) #, random_state=42)
    for i_fold, (indi_tra, indi_tes) in enumerate(skf.split(X_all, T_all)):
        X_tra, T_tra = X_all[indi_tra], T_all[indi_tra]
        X_tes, T_tes = X_all[indi_tes], T_all[indi_tes]
        
        model = CleanLabelResNet1D(model_conf, dl_conf)

        ## Training
        model.fit(X_tra, T_tra, max_epochs=5)

        ## Prediction
        pred_class_tes = model.predict_proba(X_tes).argmax(dim=-1)
        T_tes = torch.tensor(T_tes)
        acc_tes = (pred_class_tes == T_tes).float().mean()
        print(acc_tes)
    '''
        

    

    ## EOF

        
        
