#!/usr/bin/env python

"""
Predict suspicious-ripple-including group using the trained model on the binary classification task for not-ripple-including group vs reasonable-ripple-including group.
"""

import argparse
import os
import sys

sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import utils
from models.ResNet1D.ResNet1D import ResNet1D
from torch.cuda.amp import GradScaler, autocast

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-im", "--i_mouse_test", default="01", help=" ")
args = ap.parse_args()


################################################################################
## Save dir
################################################################################
LDIR = "./ripples/detect_ripples/CNN/train_n_vs_r/"
SDIR = "./ripples/detect_ripples/CNN/predict_n_s_r/test_mouse#{}/".format(
    args.i_mouse_test
)


################################################################################
## Sets tee
################################################################################
sys.stdout, sys.stderr = utils.general.tee(sys, SDIR + "out")


################################################################################
## Fixes random seed
################################################################################
utils.general.fix_seeds(seed=42, np=np, torch=torch)


################################################################################
## Configures matplotlib
################################################################################
utils.plt.configure_mpl(plt)


################################################################################
## Loads Configurations
################################################################################
GLOBAL_CONF_USED = utils.general.load(LDIR + "global_conf_used.yaml")
DL_CONF = utils.general.load(LDIR + "dataloader_conf.yaml")
MODEL_CONF = utils.general.load(LDIR + "model_conf.yaml")


################################################################################
## Adjusts Configurations
################################################################################
DL_CONF["use_classes_str"] = ["n", "s", "r"]
DL_CONF["batch_size"] = 256


################################################################################
## Prepares training
################################################################################
# reporter = utils.ml.Reporter(SDIR)
device = "cuda"
scaler = GradScaler()


################################################################################
## Main
################################################################################
n_mouse_test_str = args.i_mouse_test
i_mouse_test = int(n_mouse_test_str) - 1


################################################################################
## Initializes model
################################################################################
MODEL_CONF = utils.general.load("./models/ResNet1D/ResNet1D.yaml")
model = ResNet1D(MODEL_CONF)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs.")
    model = nn.DataParallel(model)
model = model.to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


################################################################################
## Loads weights
################################################################################
CHECKPOINTS_DIR = LDIR + "checkpoints/"
LPATH_CHECKPOINT_PTH = CHECKPOINTS_DIR + "mouse_test#0{}_epoch#{:03d}.pth".format(
    i_mouse_test + 1, DL_CONF["MAX_EPOCHS"] - 1
)
cpts = utils.general.load(LPATH_CHECKPOINT_PTH, show=True)


################################################################################
## Sets weights on the model and optimizer
################################################################################
model.load_state_dict(cpts["model_state_dict"])
# optimizer.load_state_dict(cpts["optimizer_state_dict"])

DL_CONF["i_mouse_test"] = i_mouse_test
dlf = utils.pj.DataLoaderFiller(**DL_CONF)


################################################################################
## Test
step = "Test"
dl_test = dlf.fill(step)
P_test_all = []
pred_proba_as_r_test_all = []
model.eval()
for i_batch, batch in enumerate(dl_test):
    Xb_test, Pb_test = batch  # Pb: batched peak ripple amplitude [SD]
    Xb_test, Pb_test = Xb_test.to(device), Pb_test.to(device)

    with autocast():
        logits_test = model(Xb_test)

    pred_proba_test = logits_test.softmax(dim=-1)
    pred_class_test = pred_proba_test.argmax(dim=-1)

    pred_proba_as_r_test = pred_proba_test[:, 1]

    P_test_all.append(Pb_test.detach().cpu().numpy())
    pred_proba_as_r_test_all.append(pred_proba_as_r_test.detach().cpu().numpy())

P_test_all = np.hstack(P_test_all)
pred_proba_as_r_test_all = np.hstack(pred_proba_as_r_test_all)


## Saves
utils.general.save(P_test_all, SDIR + "P_test_all.npy", show=True)
utils.general.save(
    pred_proba_as_r_test_all, SDIR + "pred_proba_as_r_test_all.npy", show=True
)


# ## Plots
# x = P_test_all
# y = pred_proba_as_r_test_all
# fig, ax = plt.subplots()
# ax.set_title("mouse#{}; {}".format(i_mouse_test, step))
# ax.set_ylabel("Ripple probability [a.u.]")
# ax.set_xlabel("Ripple peak magnitude [SD]")
# ax.scatter(x, y, alpha=0.1)
# ax.set_xlim(0, 7.1)
# fig.show()


## EOF
