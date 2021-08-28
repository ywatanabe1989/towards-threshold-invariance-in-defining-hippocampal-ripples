#!/usr/bin/env python

"""
Fits a sigmoid curve on predicted scores of suspicious-ripple-including group ("s") as reasonable-ripple-including group ("r") using the trained model on the binary classification task for not-ripple-including group ("n") vs reasonable-ripple-including group ("r").
"""

import argparse
import os
import sys

sys.path.append(".")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils
from scipy.optimize import curve_fit

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument("-im", "--i_mouse_test", default="01", help=" ")
args = ap.parse_args()


################################################################################
## Functions
################################################################################
def sigmoid(x, a, b):
    y = 1.0 / (1.0 + np.exp(-b * (x - a)))
    return y


def sample(x, y, min, max, N_sample=100):
    indi_b = (min <= x) & (x < max)

    if N_sample <= indi_b.sum():
        return np.random.permutation(y[indi_b])[:N_sample]
    else:
        return np.concatenate([y[indi_b], np.nan * np.ones(N_sample - len(y[indi_b]))])


################################################################################
## Save dir
################################################################################
LDIR = "./ripples/detect_ripples/CNN/predict_n_s_r/test_mouse#{}/".format(
    args.i_mouse_test
)
SDIR = "./ripples/detect_ripples/CNN/fit_sigmoid_on_the_predicted_scores_of_s/"


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
## Loads
################################################################################
P_test_all = utils.general.load(LDIR + "P_test_all.npy")
pred_proba_as_r_test_all = utils.general.load(LDIR + "pred_proba_as_r_test_all.npy")
# to (x, y)
x = P_test_all
y = pred_proba_as_r_test_all
x[np.isnan(x)] = 0.5

################################################################################
## Fits a sigmoid curve on the original data
################################################################################
popt, pcov = curve_fit(sigmoid, x, y, maxfev=10000)
x_sigmoid = np.linspace(0.1, 10, 100)
y_sigmoid = sigmoid(x_sigmoid, *popt)

################################################################################
## Plots the original data and the sigmoid curve
################################################################################
fig, ax = plt.subplots()
ax.set_title("Test mouse#{}".format(args.i_mouse_test))
ax.set_ylabel("Ripple probability [a.u.]")
ax.set_xlabel("Ripple peak magnitude [SD]")
ax.scatter(x, y, alpha=0.01)
ax.plot(x_sigmoid, y_sigmoid, color="purple")
ax.set_title("Test Mouse#{}; a={:.2f}; b={:.2f}".format(args.i_mouse_test, *popt))
ax.set_xscale("log")
# fig.show()
# utils.general.save(fig, SDIR + "mouse#{}_fitted_sigmoid.png".format(args.i_mouse_test))
plt.close()

################################################################################
## Picks 100 samples and takes mean and std of them
################################################################################
x_min, x_max = 0.1, 10.0
x_bin = 0.1
n_x = 100
l = np.logspace(np.log10(x_min), np.log10(x_max), n_x)
u = l[1:]
l = l[:-1]
print(l[:10])
print(u[:10])


sampled = np.array([sample(x, y, l[i], u[i], N_sample=100) for i in range(len(l))])
x_axis = np.array([(l[i] + u[i]) / 2 for i in range(len(l))])  # /2; fixme
means = sampled.mean(axis=-1)
stds = sampled.std(axis=-1)
fig, ax = plt.subplots()
ax.errorbar(x_axis, means, stds)
ax.set_xscale("log")
ax.set_title("Test Mouse#{}".format(args.i_mouse_test))
ax.set_ylabel("Ripple probability [a.u.]")
ax.set_xlabel("Ripple peak magnitude [SD]")
# fig.show()
# utils.general.save(fig, SDIR + "mouse#{}_mean_std.png".format(args.i_mouse_test))
plt.close()

################################################################################
## Merge
################################################################################
fig, ax = plt.subplots()
ax.set_title("Test Mouse#{}".format(args.i_mouse_test))
ax.set_ylabel("Ripple probability [a.u.]")
ax.set_xlabel("Ripple peak magnitude [SD]")
ax.scatter(x, y, alpha=0.1, label="original")
ax.plot(
    x_sigmoid,
    y_sigmoid,
    color="purple",
    label="fitted on original data (a={:.2f}; b={:.2f})".format(*popt),
    linewidth=3,
)
ax.errorbar(
    x_axis,
    means,
    stds,
    color="orange",
    label="mean +/- std.",
    linewidth=3,
)
ax.legend()
ax.set_xlim(0, 10)
# fig.show()
utils.general.save(
    fig, SDIR + "mouse#{}_mean_std_and_fitted_sigmoid.png".format(args.i_mouse_test)
)

# SDs = x
# pred_probs = y

# bin_SD = 0.1
# _SDs = []
# _pred_probs = []
# prob_means = []
# prob_stds = []
# max_SD = 20
# x_sd = np.arange(0, max_SD, bin_SD) + bin_SD / 2
# flag, i = True, 0
# N_sample = 100
# while flag:
#     start, end = i * bin_SD, (i + 1) * bin_SD
#     indi = ((start <= SDs) & (SDs < end)).squeeze()
#     indi = np.random.permutation(np.where(indi == True)[0])[:N_sample]
#     _SDs.append(SDs[indi])
#     _pred_probs.append(pred_probs[indi])
#     prob_means.append(pred_probs[indi].mean())
#     prob_stds.append(pred_probs[indi].std())
#     i += 1
#     if max_SD <= end:
#         flag = False

# _SDs, _pred_probs = (
#     np.vstack(_SDs).reshape(-1, 1).squeeze(),
#     np.vstack(_pred_probs).reshape(-1, 1).squeeze(),
# )
# prob_means, prob_stds = np.array(prob_means), np.array(prob_stds)
# popt, pcov = curve_fit(sigmoid, _SDs, _pred_probs, maxfev=10000)
# x_sigmoid = np.linspace(0, max_SD, 1000)
# y_sigmoid = sigmoid(x_sigmoid, *popt)


# df = pd.DataFrame(
#     {
#         "x": x,
#         "logx": np.log10(x),
#         "y": y,
#     }
# )

# l = np.logspace(0, 2.0, 100) - 1
# u = l[1:]
# l = l[:-1]
# print(l[:10])
# print(u[:10])


# # df['logx'].max()

# df_plt = pd.DataFrame()

# x_axis = [(l[i] + u[i]) / 2 for i in range(len(l))]
# means = [df["y"][(l[i] < df["x"]) & (df["x"] < u[i])].mean() for i in range(len(l))]
# stds = [df["y"][(l[i] < df["x"]) & (df["x"] < u[i])].std() for i in range(len(l))]
# fig, ax = plt.subplots()
# ax.errorbar(x_axis, means, yerr=stds)
# fig.show()
# df["l"] = l
# df["u"] = u

# # ################################################################################
# # ## Loads Configurations
# # ################################################################################
# # GLOBAL_CONF_USED = utils.general.load(LDIR + "global_conf_used.yaml")
# # DL_CONF = utils.general.load(LDIR + "dataloader_conf.yaml")
# # MODEL_CONF = utils.general.load(LDIR + "model_conf.yaml")


# # ################################################################################
# # ## Adjusts Configurations
# # ################################################################################
# # DL_CONF["use_classes_str"] = ["s"]
# # DL_CONF["batch_size"] = 256


# # ################################################################################
# # ## Prepares training
# # ################################################################################
# # # reporter = utils.ml.Reporter(SDIR)
# # device = "cuda"
# # scaler = GradScaler()


# # ################################################################################
# # ## Main
# # ################################################################################
# # n_mouse_test_str = args.i_mouse_test
# # i_mouse_test = int(n_mouse_test_str) - 1


# # ################################################################################
# # ## Initializes model
# # ################################################################################
# # MODEL_CONF = utils.general.load("./models/ResNet1D/ResNet1D.yaml")
# # model = ResNet1D(MODEL_CONF)
# # if torch.cuda.device_count() > 1:
# #     print("Let's use", torch.cuda.device_count(), "GPUs!")
# #     model = nn.DataParallel(model)
# # model = model.to(device)
# # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# # ################################################################################
# # ## Loads weights
# # ################################################################################
# # CHECKPOINTS_DIR = LDIR + "checkpoints/"
# # LPATH_CHECKPOINT_PTH = CHECKPOINTS_DIR + "mouse_test#0{}_epoch#{:03d}.pth".format(
# #     i_mouse_test + 1, DL_CONF["MAX_EPOCHS"] - 1
# # )
# # cpts = utils.general.load(LPATH_CHECKPOINT_PTH, show=True)


# # ################################################################################
# # ## Sets weights on the model and optimizer
# # ################################################################################
# # model.load_state_dict(cpts["model_state_dict"])
# # optimizer.load_state_dict(cpts["optimizer_state_dict"])

# # DL_CONF["i_mouse_test"] = i_mouse_test
# # dlf = utils.pj.DataLoaderFiller(**DL_CONF)


# # ################################################################################
# # ## Test
# # step = "Test"
# # dl_test = dlf.fill(step)
# # P_test_all = []
# # pred_proba_as_r_test_all = []
# # model.eval()
# # for i_batch, batch in enumerate(dl_test):
# #     Xb_test, Pb_test = batch  # Pb: batched peak ripple amplitude [SD]
# #     Xb_test, Pb_test = Xb_test.to(device), Pb_test.to(device)

# #     with autocast():
# #         logits_test = model(Xb_test)

# #     pred_proba_test = logits_test.softmax(dim=-1)
# #     pred_class_test = pred_proba_test.argmax(dim=-1)

# #     pred_proba_as_r_test = pred_proba_test[:, 1]

# #     P_test_all.append(Pb_test.detach().cpu().numpy())
# #     pred_proba_as_r_test_all.append(pred_proba_as_r_test.detach().cpu().numpy())

# # P_test_all = np.hstack(P_test_all)
# # pred_proba_as_r_test_all = np.hstack(pred_proba_as_r_test_all)


# # ## Saves
# # utils.general.save(P_test_all, SDIR + "P_test_all.npy", show=True)
# # utils.general.save(P_test_all, SDIR + "pred_proba_as_r_test_all.npy", show=True)


# # ## Plots
# # x = P_test_all
# # y = pred_proba_as_r_test_all
# # fig, ax = plt.subplots()
# # ax.set_title("mouse#{}; {}".format(i_mouse_test, step))
# # ax.set_ylabel("Ripple probability [a.u.]")
# # ax.set_xlabel("Ripple peak magnitude [SD]")
# # ax.scatter(x, y, alpha=0.1)
# # ax.set_xlim(0, 7.1)
# # fig.show()


# ## EOF
