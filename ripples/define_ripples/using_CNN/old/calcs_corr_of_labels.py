#!/usr/bin/env python
import argparse
import sys

sys.path.append(".")
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import utils

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument(
    "-nm", "--n_mouse", default="01", choices=["01", "02", "03", "04", "05"], help=" "
)
args = ap.parse_args()


## configure matplotlib
utils.plt.configure_mpl(plt)

################################################################################
## FPATHs
################################################################################
LPATH_HIPPO_LFP_NPY_LIST = utils.general.load(
    "./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt"
)
LPATH_HIPPO_LFP_NPY_LIST_MICE = utils.general.grep(
    LPATH_HIPPO_LFP_NPY_LIST, args.n_mouse
)[1]

dataset_keys = ["D0{}-".format(i + 1) for i in range(5)]
dataset_keys[int(args.n_mouse) - 1] = dataset_keys[int(args.n_mouse) - 1].replace(
    "-", "+"
)
print("\nTarget Mouse: {}\nDataset Keys: {}\n".format(args.n_mouse, dataset_keys))
SDIR = "./ripples/define_ripples/using_CNN/calcs_corr_of_labels/"

################################################################################
## Loads
################################################################################
labels_dict = utils.general.listed_dict(dataset_keys)
pred_probas_dict = utils.general.listed_dict(dataset_keys)
for dk in dataset_keys:
    _, rips_df_list_CNN = utils.pj.load.lfps_rips_sec(
        LPATH_HIPPO_LFP_NPY_LIST_MICE, rip_sec_ver="CNN_labeled/{}".format(dk)
    )
    rips_df = pd.concat(rips_df_list_CNN)
    labels_dict[dk] = rips_df["are_ripple_CNN"]
    pred_probas_dict[dk] = rips_df["pred_probas_ripple_CNN"]


################################################################################
## Label concordance rate
################################################################################
five_diag_arr = np.diag([1 for _ in range(5)]).astype(np.float64)
label_concordance_rates_df = pd.DataFrame(
    data=five_diag_arr, columns=dataset_keys, index=dataset_keys
)

label_concordance_rates_df = label_concordance_rates_df[::-1]

for i_combi, combi in enumerate(combinations(dataset_keys, 2)):
    concordance_rate = np.round(
        (labels_dict[combi[0]] == labels_dict[combi[1]]).mean(), 2
    )
    label_concordance_rates_df.loc[combi[0], combi[1]] = concordance_rate
print("\nLabels Concordance Rates:\n{}\n".format(label_concordance_rates_df))

fig, ax = plt.subplots()
ax.set_aspect(1)
fig = sns.heatmap(
    label_concordance_rates_df,
    mask=label_concordance_rates_df == 0,
    cmap="Blues",
    annot=True,
    ax=ax,
    vmin=0.0,
    vmax=1.0,
)
# fig.figure.show()


utils.general.save(
    fig,
    SDIR + "label_concordance_rates/images/mouse_#{}.png".format(args.n_mouse),
)

utils.general.save(
    label_concordance_rates_df,
    SDIR + "label_concordance_rates/matrices/mouse_#{}.csv".format(args.n_mouse),
)


################################################################################
## Correlation of Predicted Probabilities for Ripples
################################################################################
corr_pred_probas_df = pd.DataFrame(
    data=five_diag_arr, columns=dataset_keys, index=dataset_keys
)
for i_combi, combi in enumerate(combinations(dataset_keys, 2)):
    corr = np.round(
        np.corrcoef(pred_probas_dict[combi[0]], pred_probas_dict[combi[1]]), 2
    )[0, 1]
    corr_pred_probas_df.loc[combi[0], combi[1]] = corr

corr_pred_probas_df = corr_pred_probas_df[::-1]

print(
    "\nCorrelation Coefficients of Predicted Probabilities for Ripples by CNN:\n{}\n".format(
        corr_pred_probas_df
    )
)

fig, ax = plt.subplots()
ax.set_aspect(1)
fig = sns.heatmap(
    label_concordance_rates_df,
    mask=label_concordance_rates_df == 0,
    cmap="Blues",
    annot=True,
    ax=ax,
    vmin=0.0,
    vmax=1.0,
)

# fig.figure.show()

utils.general.save(
    fig, SDIR + "corr_coef_pred_proba/images/mouse_#{}.png".format(args.n_mouse)
)

utils.general.save(
    corr_pred_probas_df,
    SDIR + "corr_coef_pred_proba/matrices/mouse_#{}.csv".format(args.n_mouse),
)


## EOF
