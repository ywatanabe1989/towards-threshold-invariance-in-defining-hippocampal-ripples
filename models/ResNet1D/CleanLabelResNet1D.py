#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import (autocast,
                            GradScaler)

from sklearn.base import BaseEstimator
from sklearn.utils import shuffle

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
    def __init__(self, cl_conf):
        self.cl_conf = cl_conf
        self.resnet1d_conf = cl_conf['ResNet1D']
        self.dl_conf = cl_conf['dataloader']

        ## Model
        self.resnet1d = ResNet1D(self.resnet1d_conf)
        self.resnet1d = nn.DataParallel(self.resnet1d)
        self.resnet1d = self.resnet1d.to(self.resnet1d_conf['device'])

        ## Training
        self.softmax = nn.Softmax(dim=-1)
        self.xentropy_criterion = nn.CrossEntropyLoss(reduction='none')
        self.loss_balancer = LossBalancer(len(self.resnet1d_conf['LABELS']),
                                          self.dl_conf['batch_size'],
                                          )
        self.loss_tra = []
        self.log_interval_batch = 50

        
    def fit(self, X, T, sample_weight=None):
        '''This method adheres to sklearn's "fit(X, y)" format for compatibility with scikit-learn.
        All inputs should be numpy arrays (not PyTorch Tensors).
        train_idx is not X, but instead a list of indices for X (and y if train_labels is None).
        '''

        X, T = shuffle(X, T)
        
        ds_tra = torch.utils.data.TensorDataset(torch.FloatTensor(X),
                                                torch.LongTensor(T))
        
        dl_tra = torch.utils.data.DataLoader(dataset=ds_tra,
                                             **self.dl_conf,
                                             shuffle=True,
                                             drop_last=True,
                                  )    
        
        optimizer = Ranger(self.resnet1d.parameters(), lr=self.cl_conf['lr'])

        scaler = GradScaler()

        # Train for self.epochs epochs
        self.resnet1d.train()        
        for i_epoch in range(1, self.cl_conf['max_epochs'] + 1):
            for i_batch, batch in enumerate(dl_tra):
                Xb, Tb = batch

                Xb = Xb.to(self.resnet1d_conf['device'])
                Tb = Tb.to(self.resnet1d_conf['device'])
                    
                optimizer.zero_grad()
                
                with autocast():
                    y = self.resnet1d(Xb)
                    loss = self.xentropy_criterion(y, Tb.long())
                    loss = self.loss_balancer(loss, Tb, i_epoch, train=True).mean()
                    self.loss_tra.append(loss.item())
                    
                scaler.scale(loss).backward() # loss.backward()
                scaler.step(optimizer) # optimizer.step()
                scaler.update()
                
                if i_batch % self.log_interval_batch == 0:
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
    

    def predict_proba(self, X):
        ## Adds shadow rows for the multi GPU model
        residue = len(X) % self.cl_conf['dataloader']['batch_size']
        n_added = self.cl_conf['dataloader']['batch_size'] - residue
        shadow_rows = np.zeros([n_added, X.shape[1]])
        X = np.concatenate([X, shadow_rows], axis=0)
        # print('{} shadow rows were aded.'.format(n_added))    
        # check
        residue = len(X) % self.cl_conf['dataloader']['batch_size']
        assert residue == 0

        
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

            with autocast():
                # print(Xb.shape)
                y = self.resnet1d(Xb)
                outs.append(y.detach().cpu())

        outs = torch.cat(outs, dim=0)[:-n_added] # Excludes the results of shadow rows
        pred = self.softmax(outs.float())

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

    import utils.general as ug
    from utils.Reporter import Reporter


    ## Fix seeds
    ug.fix_seeds(random=random, np=np)

    
    ## Data
    mnist = load_digits(n_class=2)
    X_all, T_all = mnist.data, mnist.target # float64
    labels= ['0', '1']
    bs, n_chs, seq_len = 16, 1, X_all.shape[-1]

    ################################################################################    
    ## Model
    ################################################################################    
    cl_conf = ug.load('./models/ResNet1D/CleanLabelResNet1D.yaml')
    cl_conf['ResNet1D']['SEQ_LEN'] = X_all.shape[-1]
    cl_conf['ResNet1D']['LABELS'] = labels
    cl_conf['max_epochs'] = 2
    cl_conf['dataloader']['batch_size'] = 32
    
    model = CleanLabelResNet1D(cl_conf)

    ################################################################################    
    ## Main; just trains the model and predicts test data as usual
    ################################################################################
    ## https://github.com/cgnorthcutt/cleanlab/blob/master/cleanlab/latent_estimation.py
    ## estimate_confident_joint_and_cv_pred_proba()
    
    ## Paramters
    N_FOLDS = 5
    reporter = Reporter(sdir='/tmp/')
    skf = StratifiedKFold(n_splits=N_FOLDS)
    N_CLASSES = len(np.unique(T_all))
    psx = np.zeros((len(T_all), N_CLASSES))
    for i_fold, (indi_tra, indi_tes) in enumerate(skf.split(X_all, T_all)):
        X_tra, T_tra = X_all[indi_tra], T_all[indi_tra]
        X_tes, T_tes = X_all[indi_tes], T_all[indi_tes]

        ## Instantiates a Model
        model = CleanLabelResNet1D(cl_conf)

        ## Training
        model.fit(X_tra, T_tra)
        
        ## Prediction
        pred_proba_tes_fold = model.predict_proba(X_tes)
        pred_class_tes_fold = pred_proba_tes_fold.argmax(dim=-1)
        len_tes_dropped = len(pred_proba_tes_fold)
        T_tes_fold = torch.tensor(T_tes)
        
        reporter.calc_metrics(T_tes_fold,
                              pred_class_tes_fold,
                              pred_proba_tes_fold,
                              labels=cl_conf['ResNet1D']['LABELS'],
                              i_fold=i_fold,
                              )
        ## to the buffer
        psx[indi_tes] = pred_proba_tes_fold

    are_errors = get_noise_indices(T_all,
                                   psx,
                                   inverse_noise_matrix=None,
                                   prune_method='prune_by_noise_rate',
                                   n_jobs=20,
                                   )
    are_ones = psx[:, 1]

    print('\nLabel Errors Indice:\n{}\n'.format(are_errors))

    reporter.summarize()
    others_dict = {'are_errors.npy': are_errors,
                   'are_ones.npy': are_ones,
                   }
    reporter.save(others_dict=others_dict)

    
    '''
    ################################################################################
    ## Confident Learning using cleanlab
    ################################################################################    
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

    # ## imshow
    # indi = np.where(np.array(are_errors) == True)[0]
    # for ii in indi:
    #     Xi, Ti = X_all[ii], T_all[ii]
    #     plt.imshow(Xi.reshape(8,8))
    #     plt.title(Ti)
    #     plt.show()
    '''    
    
    ## EOF
