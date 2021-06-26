#!/usr/bin/env python

"""
Trains binary classifier for not-ripple-including group vs reasonable-ripple-including group.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import utils
from models.ResNet1D.ResNet1D import ResNet1D
from torch.cuda.amp import GradScaler, autocast

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument(
    "-im",
    "--i_mouse_test",
    default="01",
    choices=["01", "02", "03", "04", "05"],
    help=" ",
)
args = ap.parse_args()


################################################################################
## Save dir
################################################################################
SDIR = "/tmp/"

################################################################################
## Sets tee
################################################################################
# sys.stdout, sys.stderr = utils.general.tee(sys)
sys.stdout, sys.stderr = utils.general.tee(sys, SDIR)


################################################################################
## Fixes random seed
################################################################################
utils.general.fix_seeds(seed=42, np=np)


################################################################################
## Configures matplotlib
################################################################################
utils.plt.configure_mpl(plt)


################################################################################
## Initializes Dataloader
################################################################################
kwargs_dl = {
    "i_mouse_test": 0,
    "batch_size": 256,
    "num_workers": 10,
    "do_under_sampling": True,
    "use_classes_str": ["n", "r"],
    "samp_rate": 1000,
    "window_size_pts": 400,
    "use_random_start": True,
    "lower_SD_thres_for_reasonable_ripple": 7,
}
dlf = utils.pj.DataLoaderFiller(**kwargs_dl)


################################################################################
## Initializes model
################################################################################
resnet_config = utils.general.load("./models/ResNet1D/ResNet1D.yaml")
model = ResNet1D(resnet_config).cuda()


################################################################################
## Prepares training
################################################################################
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
# scaler = GradScaler()
lc_logger = utils.ml.LearningCurveLogger()
reporter = utils.ml.Reporter(SDIR)
device = "cuda"

################################################################################
## Main
################################################################################
i_global = 0
for i_epoch in range(1):
    dl_tra = dlf.fill("Training")
    dl_tes = dlf.fill("Test")

    ## Training
    for i_batch, batch in enumerate(dl_tra):
        optimizer.zero_grad()

        Xb_tra, _Pb_tra = batch  # Pb: batched peak ripple amplitude [SD]

        loss = utils.pj.base_step(
            model,
            optimizer,
            "Training",
            batch,
            device,
            lc_logger,
            int(args.i_mouse_test) - 1,
            i_epoch,
            i_batch,
            i_global,
        )

        # ## To Target labels
        # # 0: 'n'/not-ripple-including group
        # # 1: 'r'/reasonable one-ripple-including group
        # Tb_tra = (~_Pb_tra.isnan()).long()
        # Xb_tra, Tb_tra = Xb_tra.cuda(), Tb_tra.cuda()

        # with autocast():
        #     logits_tra = model(Xb_tra)
        #     loss = loss_fn(logits_tra, Tb_tra)

        # pred_proba_tra = softmax(logits_tra)
        # pred_class_tra = pred_proba_tra.argmax(dim=-1)

        i_global += 1

    ## Test
    T_tes_all, pred_class_tes_all, pred_proba_tes_all = [], [], []
    for i_batch, batch in enumerate(dl_tes):
        Xb_tes, _Pb_tes = batch  # Pb: batched peak ripple amplitude [SD]
        _ = utils.pj.base_step(
            model,
            optimizer,
            "Test",
            batch,
            device,
            lc_logger,
            int(args.i_mouse_test) - 1,
            i_epoch,
            i_batch,
            i_global,
        )

        # ## To Target labels
        # # 0: 'n'/not-ripple-including group
        # # 1: 'r'/reasonable one-ripple-including group
        # Tb_tes = (~_Pb_tes.isnan()).long()
        # Xb_tes, Tb_tes = Xb_tes.cuda(), Tb_tes.cuda()

        # with autocast():
        #     logits_tes = model(Xb_tes)

        # pred_proba_tes = softmax(logits_tes)
        # pred_class_tes = pred_proba_tes.argmax(dim=-1)

        # T_tes_all.append(Tb_tes.detach().cpu().numpy())
        # pred_proba_tes_all.append(pred_proba_tes.detach().cpu().numpy())
        # pred_class_tes_all.append(pred_class_tes.detach().cpu().numpy())

    T_tes_all = np.hstack(T_tes_all)
    pred_class_tes_all = np.hstack(pred_class_tes_all)
    pred_proba_tes_all = np.vstack(pred_proba_tes_all)

    ## Metrics
    reporter.calc_metrics(
        T_tes_all,
        pred_class_tes_all,
        pred_proba_tes_all,
        labels=["n", "r"],
        i_fold=dlf.kwargs["i_mouse_test"],
    )

reporter.summarize()
