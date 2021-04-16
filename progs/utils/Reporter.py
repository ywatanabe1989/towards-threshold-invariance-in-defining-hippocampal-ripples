#!/usr/bin/env python

from natsort import natsorted
import numpy as np
from sklearn.metrics import (confusion_matrix,
                             classification_report,
                             roc_auc_score,
                             )
import pandas as pd
import torch
import yaml


from utils.general import (mv_to_tmp,
                           torch_to_arr,
                           gen_timestamp,
                           save_listed_scalars_as_csv,
                           save_listed_dfs_as_csv,
                           connect_str_list_with_hyphens,
                           )
from utils.ml import plot_cm

class Reporter():
    '''model_name and window_size_sec [sec] is required for instantiation.
       Author name is fetched from environmental viriable "USER" (i.g., ywatanabe).
       Summaried data will be saved under
       '/storage/data/EEG/EEG_DiagnosisFromRawSignals/for_0416_AMED/\
        "USER"/ML/"model_name"_WS-"window_size_sec"sec_"timestamp"/'

       labels:
    '''
    def __init__(self, model_name, window_size_sec, load_diags_str):
        ## Make save dir
        clf_combi = connect_str_list_with_hyphens(natsorted(set(load_diags_str)))
        AUTHOR = os.environ['USER']
        self.sdir = '/storage/data/EEG/EEG_DiagnosisFromRawSignals/for_0416_AMED/\
                     {}/ML/{}/_{}_WS-{}sec_{}/'\
            .format(AUTHOR, clf_combi, model_name,
                    window_size_sec, gen_timestamp()).replace(' ', '')

        self.conf_mats_folds = []
        self.clf_reports_folds = []
        self.roc_aucs_folds = []
        

    def calc_metrics(self, true_class, pred_class, pred_proba, labels=None, i_fold=None):
        '''Calculates ACC, Confusion Matrix, Classification Report, and ROC-AUC score.'''
        self.labels = labels
        
        true_class, pred_class, pred_proba = \
            torch_to_arr(true_class), torch_to_arr(pred_class), torch_to_arr(pred_proba)

        ## ACC ##
        acc = (true_class.reshape(-1) == pred_class.reshape(-1)).mean()
        print('\nACC in fold#{} was {:.3f}\n'.format(i_fold, acc))

        ## Confusion Matrix ##
        conf_mat = confusion_matrix(true_class, pred_class)
        conf_mat = pd.DataFrame(data=conf_mat, columns=labels).set_index(pd.Series(list(labels)))
        print('\nConfusion Matrix in fold#{}: \n{}\n'.format(i_fold, conf_mat))

        ## Classification Report ##
        clf_report = \
            pd.DataFrame(
            classification_report(true_class, pred_class,
                                  target_names=labels,
                                  output_dict=True,
                                  )
            ).round(3)
        # rename 'support' to 'sample size'
        clf_report['index'] = clf_report.index
        clf_report.loc['support', 'index'] = 'sample size'
        clf_report.set_index('index', drop=True, inplace=True)
        clf_report.index.name = None
        print('\nClassification Report in fold#{}: \n{}\n'.format(i_fold, clf_report))

        ## ROC-AUC score ##
        n_classes = len(labels)
        if n_classes == 2:
            multi_class = 'raise'
            pred_proba = pred_proba[:, 1]
        else:
            multi_class = 'ovo'
        roc_auc = roc_auc_score(true_class, pred_proba, 
                                multi_class=multi_class,
                                average='macro',
                                ) # fixme; multi_class for 2-class classification problem
        '''
        'ovo' stands for One-vs-one. This option makes it compute the average AUC of
        all possible pairwise combinations of classes. Insensitive to class imbalance
        when average == 'macro'.
        '''
        print('\nROC_AUC in fold#{} was {:.3f}\n'.format(i_fold, roc_auc))

        self.conf_mats_folds.append(conf_mat)
        self.clf_reports_folds.append(clf_report)
        self.roc_aucs_folds.append(roc_auc)

        # return acc, conf_mat, clf_report, roc_auc

    def summarize(self,):
        ## Summarize each fold's metirics
        self.conf_mat_cv_sum = summarize_dfs(self.conf_mats_folds, method='sum')
        self.clf_report_cv_mean, self.clf_report_cv_std = \
            summarize_dfs(self.clf_reports_folds, method='mean')
        self.roc_auc_cv_mean, self.roc_auc_cv_std = take_mean_and_std(self.roc_aucs_folds)

        self.num_folds = len(self.conf_mats_folds)

        print('\n --- {}-fold CV overall metrics --- \n'.format(self.num_folds))
        print('\nConfusion Matrix (Test; sum; num. folds={})\n{}\n'\
              .format(self.num_folds, self.conf_mat_cv_sum))
        print('\nClassification Report (Test; mean; num. folds={})\n{}\n'\
              .format(self.num_folds, self.clf_report_cv_mean))
        print('\nClassification Report (Test; std; num. folds={})\n{}\n'\
              .format(self.num_folds, self.clf_report_cv_std))
        print('\nROC AUC Score: {} +/- {} (mean +/- std.; n={})\n'\
              .format(self.roc_auc_cv_mean, self.roc_auc_cv_std, self.num_folds))

    def save(self, meta_dict=None, labels=None):
        os.makedirs(self.sdir, exist_ok=True)        
        
        # labels = list(dlpacker.diag_str_to_int_dict.keys())
        # {'HV': 0, 'DLB': 1, 'iNPH': 2, 'AD': 3}

        if meta_dict is not None:
            spath_meta_yaml = self.sdir + 'meta.yaml'
            with open(spath_meta_yaml, 'w') as f:
                yaml.dump(meta_dict, f)
            print('Saved to: {}'.format(spath_meta_yaml))
        
        
        ## Confusion Matrix
        # Values
        conf_mats_cat = [self.conf_mat_cv_sum] + self.conf_mats_folds
        indi_suffix_cat = ['{}-fold CV SUM'.format(self.num_folds)] \
                        + ['fold#{}'.format(i) for i in range(self.num_folds)]
        save_listed_dfs_as_csv(conf_mats_cat, self.sdir + 'conf_mats.csv',
                               indi_suffix=indi_suffix_cat, overwrite=True)
        # Figures; fixme; each fold
        spath = self.sdir + 'conf_mat_overall_sum.png'
        plot_cm(self.conf_mat_cv_sum, labels=labels, spath=spath)


        ## Classification Matrix
        clf_reports_cat = [self.clf_report_cv_mean] \
                        + [self.clf_report_cv_std] \
                        + self.clf_reports_folds
        indi_suffix_cat = ['{}-fold CV mean'.format(self.num_folds)] \
                        + ['{}-fold CV std.'.format(self.num_folds)] \
                        + ['fold#{}'.format(i) for i in range(self.num_folds)]
        save_listed_dfs_as_csv(clf_reports_cat, self.sdir + 'clf_reports.csv',
                               indi_suffix=indi_suffix_cat, overwrite=True)


        ## ROC-AUC
        roc_aucs_cat = [self.roc_auc_cv_mean, self.roc_auc_cv_std] + self.roc_aucs_folds
        indi_suffix_cat = ['{}-fold CV mean'.format(self.num_folds)] \
                        + ['{}-fold CV std.'.format(self.num_folds)] \
                        + ['fold#{}'.format(i) for i in range(self.num_folds)]
        save_listed_scalars_as_csv(roc_aucs_cat, self.sdir + 'roc-auc.csv', column_name='ROC-AUC',
                                   indi_suffix=indi_suffix_cat, overwrite=True)


def take_mean_and_std(obj_list, n_round=3):
    arr = np.array(obj_list)
    return arr.mean(axis=0).round(n_round), arr.std(axis=0).round(n_round)

def summarize_dfs(dfs_list, method='sum', n_round=3):
    df_zero = 0*dfs_list[0].copy() # get the table format
    
    if method == 'sum':
        return df_zero + np.array(dfs_list).sum(axis=0).round(n_round)

    if method == 'mean':
        arr = np.array(dfs_list)
        df_mean = arr.mean(axis=0).round(n_round)
        df_std = arr.std(axis=0).round(n_round)
        return df_zero + df_mean, df_zero + df_std

