#!/usr/bin/env python3


import skimage
import torch.nn as nn
import utils
from bisect import bisect_left
from models.ResNet1D.ResNet1D import ResNet1D

from sklearn.metrics import balanced_accuracy_score


################################################################################
## Functions
################################################################################


def counts_n_ripple_candi_in_each_seg(lfp, rip_sec, lfp_start_sec):
    ################################################################################
    ## Bins the signal into 400-ms segments, which includes into one-candidates including-LFP
    ################################################################################
    window_size_pts = 400

    ##############################################
    ## Counts ripple candidates in each segment
    ##############################################
    # time points
    rip_filled = utils.pj.fill_rip_sec(lfp, rip_sec, samp_rate)

    ## Slices LFP
    the_4th_last_rip_end_pts = int(rip_filled.iloc[-4]["end_sec"] * samp_rate)
    lfp = lfp[:the_4th_last_rip_end_pts]

    segs = skimage.util.view_as_windows(
        lfp.squeeze(),
        window_shape=(window_size_pts,),
        step=window_size_pts,
    )
    segs_start_sec = (
        np.array([window_size_pts * i for i in range(len(segs))]) + 1e-10
    ) / samp_rate + lfp_start_sec
    segs_end_sec = segs_start_sec + window_size_pts / samp_rate

    the_1st_indi = np.array(
        [
            bisect_left(rip_filled["start_sec"].values, segs_start_sec[i]) - 1
            for i in range(len(segs))
        ]
    )  # on the starting points
    the_2nd_indi = the_1st_indi + 1
    the_3rd_indi = the_1st_indi + 2

    the_1st_filled_rips = rip_filled.iloc[the_1st_indi]  # on the starting points
    the_2nd_filled_rips = rip_filled.iloc[the_2nd_indi]
    the_3rd_filled_rips = rip_filled.iloc[the_3rd_indi]

    ## Conditions
    are_the_1st_over_the_slice_end = segs_end_sec < the_1st_filled_rips["end_sec"]
    are_the_2nd_over_the_slice_end = segs_end_sec < the_2nd_filled_rips["end_sec"]
    are_the_3rd_over_the_slice_end = segs_end_sec < the_3rd_filled_rips["end_sec"]

    are_just_one_candidate_included = np.vstack(
        [
            ~the_1st_filled_rips["is_candidate"],
            ~are_the_1st_over_the_slice_end,
            the_2nd_filled_rips["is_candidate"],
            ~are_the_2nd_over_the_slice_end,
            ~the_3rd_filled_rips["is_candidate"],
            are_the_3rd_over_the_slice_end,
        ]
    ).all(axis=0)

    are_no_candidates_included = np.vstack(
        [
            ~the_1st_filled_rips["is_candidate"],
            are_the_1st_over_the_slice_end,
        ]
    ).all(axis=0)

    n_ripple_candi = np.nan * np.zeros(len(segs))
    n_ripple_candi[are_just_one_candidate_included] = 1
    n_ripple_candi[are_no_candidates_included] = 0

    return segs, n_ripple_candi


################################################################################
## Parameters
################################################################################
samp_rate = 1000


################################################################################
## Loads signal
################################################################################
# fpath = "./data/okada/D.npy"  # "./data/demo_hipp_lfp_1d.npy"
fpath = "./data/okada/01/day1/split/LFP_MEP_1kHz_npy/orig/tt1-3_fp16.npy"
# fpath = "/tmp/fake.npy"  # "./data/demo_hipp_lfp_1d.npy"
"""
hipp_lfp_1d = np.random.rand(int(1e6)).squeeze().astype(np.float32)
utils.general.save(hipp_lfp_1d, fpath)
"""
hipp_lfp_1d = np.load(fpath).squeeze().astype(np.float32)
lfp_all = hipp_lfp_1d[:, np.newaxis]


################################################################################
## Use only a part of signal for the computational reason
################################################################################
i_lfp = 0  # for searching a demo chunk
lfp_start_sec = i_lfp * 3600
lfp_end_sec = (i_lfp + 1) * 3600
lfp_step_sec = float(1 / samp_rate)
lfp_time_x = np.arange(lfp_start_sec, lfp_end_sec, lfp_step_sec).round(3)
lfp = lfp_all[lfp_start_sec * samp_rate : lfp_end_sec * samp_rate]


################################################################################
## Detects Ripple Candidates
################################################################################
lfp_h = len(lfp) / samp_rate / 60 / 60
print(f"\nDetecting ripples from {fpath} (Length: {lfp_h:.1f}h)\n".format())

lo_hz_ripple = utils.general.load("./conf/global.yaml")["RIPPLE_CANDI_LIM_HZ"][0]
hi_hz_ripple = utils.general.load("./conf/global.yaml")["RIPPLE_CANDI_LIM_HZ"][1]
_, _, rip_sec = utils.pj.define_ripple_candidates(
    lfp_time_x,
    lfp,
    samp_rate,
    lo_hz=lo_hz_ripple,
    hi_hz=hi_hz_ripple,
    zscore_threshold=1,
)


################################################################################
## Determines events to be processed in our CNN
################################################################################
(
    segs,
    n_ripple_candi,
) = counts_n_ripple_candi_in_each_seg(lfp, rip_sec, lfp_start_sec)


################################################################################
## Loads the trained model
################################################################################
MODEL_CONF = utils.general.load("./models/ResNet1D/ResNet1D.yaml")
model = ResNet1D(MODEL_CONF)
model = model.to("cuda")
model.eval()
checkpoints = utils.general.load(
    "./ripples/detect_ripples/CNN/train_FvsT/checkpoints/mouse_test#01_epoch#000.pth"
)
model.load_state_dict(
    utils.general.cvt_multi2single_model_state_dict(checkpoints["model_state_dict"])
)

################################################################################
## Estimates the ripple probabilities
################################################################################
softmax = nn.Softmax(dim=-1)
batch_size = 16
labels = {0: "F", 1: "T"}

for i in range(1000):
    """
    i = 3
    """
    ## Samples
    start = i * batch_size
    end = (i + 1) * batch_size
    Xb = segs[start:end]
    Nb = n_ripple_candi[start:end]

    ## NN Outputs
    logits = model(torch.tensor(Xb).cuda())
    pred_proba = softmax(logits)
    pred_class = pred_proba.argmax(dim=-1)  # 0: T, 1: F
    print(i, Nb)

    # print(pred_class)
    # print(Nb)
    # print(Nb.sum())
    if Nb.sum():
        print(softmax(pred_proba[Nb]).argmax(dim=-1).cpu().numpy())

    # plt.plot(Xb)
    # plt.show()

fig, ax = plt.subplots()
ax.plot(Xb.reshape(-1))
for i_s, s in enumerate(Nb.astype(str)):
    s = "0" if s == "0.0" else s
    s = "1" if s == "1.0" else s

    ax.text(i_s * 400 + 180, ax.get_ylim()[1] - 10, s)

ax.vlines(
    [i * 400 for i in range(len(Nb))],
    *ax.get_ylim(),
    linestyles="dashed",
    color="gray",
    alpha=0.5,
)

fig.show()

# bACC = balanced_accuracy_score(
#     utils.general.torch_to_arr(Tb.squeeze()),
#     utils.general.torch_to_arr(pred_class.squeeze()),
# )

## Plots the results

## EOF
