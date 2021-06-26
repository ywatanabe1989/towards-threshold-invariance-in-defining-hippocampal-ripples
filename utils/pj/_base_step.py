#!/usr/bin/env python

import torch
import torch.nn as nn
import utils
from sklearn.metrics import balanced_accuracy_score
from torch.cuda.amp import GradScaler, autocast

softmax = nn.Softmax(dim=-1)
loss_fn = nn.CrossEntropyLoss()
scaler = GradScaler()


def base_step(
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
    print_batch_interval=100,
    temperature=1.0,
    loss_weight=None,
):
    if step == "Training":
        model.train()
    elif (step == "Validation") or (step == "Test"):
        model.eval()
    else:
        print("step is either Training, Validation, or Test.")
        raise

    Xb, _Pb = batch  # Pb: batched peak ripple amplitude [SD]
    ## To Target labels
    # 0: 'n'/not-ripple-including group
    # 1: 'r'/reasonable one-ripple-including group
    Tb = (~_Pb.isnan()).long()
    Xb, Tb = Xb.cuda(), Tb.cuda()

    with autocast():
        logits = model(Xb)
        loss = loss_fn(logits, Tb)

    if step == "Training":
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # logits = model(Xb)
    # loss = loss_fn(logits, Tb)

    # pred_proba = softmax(logits)
    # pred_class = pred_proba.argmax(dim=-1)

    logits /= temperature

    pred_proba = softmax(logits)
    pred_class = pred_proba.argmax(dim=-1)

    bACC = balanced_accuracy_score(
        utils.general.torch_to_arr(Tb.squeeze()),
        utils.general.torch_to_arr(pred_class.squeeze()),
    )

    log_dict = {
        "loss": float(loss.detach().cpu().numpy()),
        "balanced_ACC": float(bACC),
        "pred_proba": pred_proba.detach().cpu().numpy(),
        "gt_label": Tb.detach().cpu().numpy(),
        "i_mouse_test": i_mouse_test,
        "i_epoch": i_epoch,
        "i_global": i_global,
    }
    lc_logger(log_dict, step=step)

    ## Print
    if i_batch % print_batch_interval == 0:
        print_txt = "\nstep: {}, \
                       i_batch: {}, \
                       loss: {:.3f}, \
        Balanced ACC: {:.3f}\n".format(
            step,
            i_batch,
            loss,
            bACC.mean(),
        )
        print(utils.general.squeeze_spaces(print_txt))

    return loss, lc_logger
    # return Tb, pred_proba, pred_class


## EOF
