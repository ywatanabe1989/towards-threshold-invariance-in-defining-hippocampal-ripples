#!/usr/bin/env python

"""
Trains binary classifier for not-ripple-including group vs reasonable-ripple-including group.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import utils
from models.ResNet1D.ResNet1D import ResNet1D
from torch.cuda.amp import GradScaler, autocast

################################################################################
## Save dir
################################################################################
LDIR = "./ripples/detect_ripples/CNN/train_n_vs_r/results/"
SDIR = "./ripples/detect_ripples/CNN/predict_s/results/"


################################################################################
## Sets tee
################################################################################
sys.stdout, sys.stderr = utils.general.tee(sys, SDIR + "out")


################################################################################
## Fixes random seed
################################################################################
utils.general.fix_seeds(seed=42, np=np)


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
DL_CONF["use_classes_str"] = ["s"]
DL_CONF["batch_size"] = 64

################################################################################
## Prepares training
################################################################################
# reporter = utils.ml.Reporter(SDIR)
device = "cuda"
scaler = GradScaler()

################################################################################
## Main
################################################################################
for n_mouse_test_str in ["01", "02", "03", "04", "05"]:
    # for n_mouse_test_str in ["01", "02"]:
    i_mouse_test = "01"  # delete me
    i_mouse_test = int(i_mouse_test) - 1

    lc_logger = utils.ml.LearningCurveLogger()

    ################################################################################
    ## Initializes model
    ################################################################################
    MODEL_CONF = utils.general.load("./models/ResNet1D/ResNet1D.yaml")
    model = ResNet1D(MODEL_CONF)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

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
    optimizer.load_state_dict(cpts["optimizer_state_dict"])

    DL_CONF["i_mouse_test"] = i_mouse_test
    dlf = utils.pj.DataLoaderFiller(**DL_CONF)

    ## Training
    step = "Training"
    dl_tra = dlf.fill(step)
    P_tra_all = []
    pred_proba_as_r_tra_all = []
    model.eval()
    for i_batch, batch in enumerate(dl_tra):
        """
        ## koko
        batch = next(iter(dl_tra))
        """
        Xb_tra, Pb_tra = batch  # Pb: batched peak ripple amplitude [SD]
        Xb_tra, Pb_tra = Xb_tra.to(device), Pb_tra.to(device)

        with autocast():
            logits_tra = model(Xb_tra)

        pred_proba_tra = logits_tra.softmax(dim=-1)
        pred_class_tra = pred_proba_tra.argmax(dim=-1)

        pred_proba_as_r_tra = pred_proba_tra[:, 1]

        P_tra_all.append(Pb_tra.detach().cpu().numpy())
        pred_proba_as_r_tra_all.append(pred_proba_as_r_tra.detach().cpu().numpy())

    P_tra_all = np.hstack(P_tra_all)
    pred_proba_as_r_tra_all = np.hstack(pred_proba_as_r_tra_all)

    ## Plots
    x = P_tra_all
    y = pred_proba_as_r_tra_all
    fig, ax = plt.subplots()
    ax.set_title(step)
    ax.set_ylabel("Ripple probability [a.u.]")
    ax.set_xlabel("Ripple peak magnitude [SD]")
    ax.scatter(x, y, alpha=0.1)
    ax.set_xlim(0, 7.1)
    fig.show()

    ################################################################################

    step = "Test"
    dl_test = dlf.fill(step)
    P_test_all = []
    pred_proba_as_r_test_all = []
    model.eval()
    for i_batch, batch in enumerate(dl_test):
        """
        ## koko
        batch = next(iter(dl_test))
        """
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

    ## Plots
    x = P_test_all
    y = pred_proba_as_r_test_all
    fig, ax = plt.subplots()
    ax.set_title(step)
    ax.set_ylabel("Ripple probability [a.u.]")
    ax.set_xlabel("Ripple peak magnitude [SD]")
    ax.scatter(x, y, alpha=0.1)
    ax.set_xlim(0, 7.1)
    fig.show()


#     for i_epoch in range(DL_CONF["MAX_EPOCHS"]):

#         for i_batch, batch in enumerate(dl_tra):
#             Xb_tra, _Pb_tra = batch  # Pb: batched peak ripple amplitude [SD]
#             Xb_tra, _Pb_tra = Xb_tra.to(device), _Pb_tra.to(device)
#             loss, lc_logger = utils.pj.base_step(
#                 model,
#                 optimizer,
#                 step,
#                 batch,
#                 device,
#                 lc_logger,
#                 i_mouse_test,
#                 i_epoch,
#                 i_batch,
#                 i_global,
#             )

#     # for i_epoch in range(DL_CONF["MAX_EPOCHS"]):
#     #     dl_tra = dlf.fill(step)
#     #     for i_batch, batch in enumerate(dl_tra):
#     #         optimizer.zero_grad()

#     #         Xb_tra, _Pb_tra = batch  # Pb: batched peak ripple amplitude [SD]
#     #         Xb_tra, _Pb_tra = Xb_tra.to(device), _Pb_tra.to(device)

#     #         loss, lc_logger = utils.pj.base_step(
#     #             model,
#     #             optimizer,
#     #             step,
#     #             batch,
#     #             device,
#     #             lc_logger,
#     #             i_mouse_test,
#     #             i_epoch,
#     #             i_batch,
#     #             i_global,
#     #         )

#     #         i_global += 1

#     # ## Test
#     # step = "Test"
#     # dl_tes = dlf.fill(step)
#     # for i_batch, batch in enumerate(dl_tes):
#     #     Xb_tes, _Pb_tes = batch  # Pb: batched peak ripple amplitude [SD]
#     #     Xb_tes, _Pb_tes = Xb_tes.to(device), _Pb_tes.to(device)
#     #     try:
#     #         i_epoch = i_epoch
#     #     except:
#     #         i_epoch = 0

#     #     _, lc_logger = utils.pj.base_step(
#     #         model,
#     #         optimizer,
#     #         step,
#     #         batch,
#     #         device,
#     #         lc_logger,
#     #         i_mouse_test,
#     #         i_epoch,
#     #         i_batch,
#     #         i_global,
#     #     )

#     ################################################################################
#     ## After training
#     ################################################################################

#     # #########################################
#     # ## Saves models only at the last epoch ##
#     # #########################################
#     # utils.general.save(
#     #     {
#     #         "i_mouse_test": i_mouse_test,
#     #         "i_epoch": i_epoch,
#     #         "model_state_dict": model.state_dict(),
#     #         "optimizer_state_dict": optimizer.state_dict(),
#     #     },
#     #     SDIR
#     #     + "checkpoints/mouse_test#{:02}_epoch#{:03d}.pth".format(
#     #         i_mouse_test + 1, i_epoch
#     #     ),
#     # )

#     # ####################
#     # ## Learning Curve ##
#     # ####################
#     # lc_fig = lc_logger.plot_learning_curves(
#     #     i_mouse_test=i_mouse_test,
#     #     max_epochs=DL_CONF["MAX_EPOCHS"],
#     #     window_size_sec=window_size_sec,
#     # )

#     # reporter.add(
#     #     "Learning Curve",
#     #     lc_fig,
#     #     {
#     #         "dirname": "learning_curves/",
#     #         "ext": ".png",
#     #     },
#     # )

#     #############
#     ## Metrics ##
#     #############
#     true_class_tes = np.hstack(lc_logger.dfs["Test"]["gt_label"])
#     pred_proba_tes = np.vstack(lc_logger.dfs["Test"]["pred_proba"])
#     pred_class_tes = pred_proba_tes.argmax(axis=-1)
#     labels = dlf.kwargs["use_classes_str"]

#     reporter.calc_metrics(
#         true_class_tes,
#         pred_class_tes,
#         pred_proba_tes,
#         labels=["n", "r"],
#         i_mouse_test=dlf.kwargs["i_mouse_test"],
#     )

# reporter.summarize()

# DL_CONF["i_global"] = i_global
# others_dict = {
#     "global_conf_used.yaml": GLOBAL_CONF_USED,
#     "model_conf.yaml": MODEL_CONF,
#     "dataloader_conf.yaml": DL_CONF,
# }
# reporter.save(others_dict=others_dict)

# ## EOF


# # from_check_point = False
# # if from_check_point:
# #     LPATH_CHECK_POINT_PTH = (
# #         SDIR
# #         + "checkpoints/mouse_test#{:02}_epoch#{:03d}.pth".format(
# #             i_mouse_test + 1, i_epoch
# #         )
# #     )

# # else:
