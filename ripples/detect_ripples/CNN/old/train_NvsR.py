#!/usr/bin/env python

"""
Trains binary classifier for not-ripple-including group ("n") vs reasonable-ripple-including group ("r").
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.append(".")
import utils
from models.ResNet1D.ResNet1D import ResNet1D
from modules.ranger2020 import Ranger

################################################################################
## Debug mode
################################################################################
IS_DEBUG = False
if IS_DEBUG:
    print("\n##################################################\n")
    print("!!! DEBUG MODE !!!")
    print("\n##################################################\n")

################################################################################
## Save dir
################################################################################
SDIR = "./ripples/detect_ripples/CNN/train_NvsR/"

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
# utils.plt.configure_mpl(
#     plt,
#     dpi=100,
#     figsize=(20, 7),
#     fontsize=7,
#     legendfontsize="xx-small",
#     hide_spines=True,
#     tick_size=0.8,
#     tick_width=0.2,
# )

# figscale = 4.0 / 11
# utils.plt.configure_mpl(plt, figscale=figscale)


################################################################################
## Global configuration
################################################################################
SAMP_RATE = utils.general.load("./conf/global.yaml")["DOWN_SAMPLED_SAMP_RATE"]
WINDOW_SIZE_PTS = utils.general.load("./conf/global.yaml")["WINDOW_SIZE_PTS"]
GLOBAL_CONF_USED = {
    "SAMP_RATE": SAMP_RATE,
    "WINDOW_SIZE_PTS": WINDOW_SIZE_PTS,
}


################################################################################
## Initializes Dataloader
################################################################################
DL_CONF = {
    "batch_size": 1024,
    "num_workers": 10,
    "do_under_sampling": True,
    "use_classes_str": ["n", "r"],
    "samp_rate": SAMP_RATE,
    "window_size_pts": WINDOW_SIZE_PTS,
    "use_random_start": True,
    "lower_SD_thres_for_reasonable_ripple": 1,
    "MAX_EPOCHS": 1,
    "is_debug": IS_DEBUG,
}
window_size_sec = DL_CONF["window_size_pts"] / DL_CONF["samp_rate"]

################################################################################
## Prepares training
################################################################################
reporter = utils.ml.Reporter(SDIR)
device = "cuda"

################################################################################
## Main
################################################################################
i_MICE_TEST = ["01", "02", "03", "04", "05"]
if IS_DEBUG:
    i_MICE_TEST = i_MICE_TEST[:2]

for i_mouse_test in i_MICE_TEST:
    i_mouse_test = int(i_mouse_test) - 1
    lc_logger = utils.ml.LearningCurveLogger()

    ################################################################################
    ## Initializes model
    ################################################################################
    MODEL_CONF = utils.general.load("./models/ResNet1D/ResNet1D.yaml")
    model = ResNet1D(MODEL_CONF)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs.")
        model = nn.DataParallel(model)

    model = model.to(device)
    optimizer = Ranger(model.parameters(), lr=1e-3)

    i_global = 0
    DL_CONF["i_mouse_test"] = i_mouse_test
    dlf = utils.pj.DataLoaderFiller(**DL_CONF)

    ## Training
    step = "Training"
    for i_epoch in range(DL_CONF["MAX_EPOCHS"]):
        dl_tra = dlf.fill(step)
        for i_batch, batch in enumerate(dl_tra):
            optimizer.zero_grad()

            Xb_tra, _Pb_tra = batch  # Pb: batched peak ripple amplitude [SD]
            Xb_tra, _Pb_tra = Xb_tra.to(device), _Pb_tra.to(device)

            loss, lc_logger = utils.pj.base_step(
                model,
                optimizer,
                step,
                batch,
                device,
                lc_logger,
                i_mouse_test,
                i_epoch,
                i_batch,
                i_global,
            )

            i_global += 1

            if IS_DEBUG == True and i_global == 10:
                break

    ## Test
    step = "Test"
    dl_tes = dlf.fill(step)
    for i_batch, batch in enumerate(dl_tes):
        Xb_tes, _Pb_tes = batch  # Pb: batched peak ripple amplitude [SD]
        Xb_tes, _Pb_tes = Xb_tes.to(device), _Pb_tes.to(device)
        try:
            i_epoch = i_epoch
        except:
            i_epoch = 0

        _, lc_logger = utils.pj.base_step(
            model,
            optimizer,
            step,
            batch,
            device,
            lc_logger,
            i_mouse_test,
            i_epoch,
            i_batch,
            i_global,
        )

    ################################################################################
    ## After training
    ################################################################################

    #########################################
    ## Saves models only at the last epoch ##
    #########################################
    utils.general.save(
        {
            "i_mouse_test": i_mouse_test,
            "i_epoch": i_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        SDIR
        + "checkpoints/mouse_test#{:02}_epoch#{:03d}.pth".format(
            i_mouse_test + 1, i_epoch
        ),
    )

    ####################
    ## Learning Curve ##
    ####################
    utils.plt.configure_mpl(
        plt,
        figsize=(8.7, 10),
        labelsize=8,
        fontsize=7,
        legendfontsize=6,
        tick_size=0.8,
        tick_width=0.2,
        n_xticks=4,
        n_yticks=4,
    )

    lc_fig = lc_logger.plot_learning_curves(
        plt,
        i_mouse_test=i_mouse_test,
        max_epochs=DL_CONF["MAX_EPOCHS"],
        window_size_sec=window_size_sec,
    )

    reporter.add(
        "Learning Curve",
        lc_fig,
        {
            "dirname": "learning_curves/",
            "ext": ".png",
        },
    )

    #############
    ## Metrics ##
    #############
    true_class_tes = np.hstack(lc_logger.dfs["Test"]["gt_label"])
    pred_proba_tes = np.vstack(lc_logger.dfs["Test"]["pred_proba"])
    pred_class_tes = pred_proba_tes.argmax(axis=-1)
    labels = dlf.kwargs["use_classes_str"]

    reporter.calc_metrics(
        true_class_tes,
        pred_class_tes,
        pred_proba_tes,
        labels=["R_0", "R_1"],
        i_fold=dlf.kwargs["i_mouse_test"],
    )

reporter.summarize()

del DL_CONF["i_mouse_test"]
DL_CONF["i_global"] = i_global
others_dict = {
    "global_conf_used.yaml": GLOBAL_CONF_USED,
    "model_conf.yaml": MODEL_CONF,
    "dataloader_conf.yaml": DL_CONF,
}
reporter.save(others_dict=others_dict)


"""
sshell
python3 ripples/detect_ripples/CNN/train_n_vs_r.py
"""

## EOF