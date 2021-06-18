#!/usr/bin/env python


import argparse

import utils

ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument(
    "-nm", "--n_mouse", default="01", choices=["01", "02", "03", "04", "05"], help=" "
)
ap.add_argument("-i", "--include", action="store_true", default=False, help=" ")
args = ap.parse_args()


################################################################################
## Fixes random seeds
################################################################################
utils.general.fix_seeds(seed=42, np=np, torch=torch)


################################################################################
## FPATHs
################################################################################
LPATH_HIPPO_LFP_NPY_LIST = utils.general.load(
    "./data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt"
)


utils.pj.load_mouse_lfp_rip()

# # Determines LPATH_HIPPO_LFP_NPY_LIST_MICE and dataset_key
# N_MICE_CANDIDATES = ["01", "02", "03", "04", "05"]
# i_mouse_tgt = utils.general.grep(N_MICE_CANDIDATES, args.n_mouse)[0][0]
# if args.include:
#     N_MICE = [args.n_mouse]
#     dataset_key = "D" + args.n_mouse + "+"
# if not args.include:
#     N_MICE = N_MICE_CANDIDATES.copy()
#     N_MICE.pop(i_mouse_tgt)
#     dataset_key = "D" + args.n_mouse + "-"

# LPATH_HIPPO_LFP_NPY_LIST_MICE = list(
#     np.hstack(
#         [
#             utils.general.grep(LPATH_HIPPO_LFP_NPY_LIST, nm)[1]
#             for nm in N_MICE
#         ]
#     )
# )
# print("Indice of mice to load: {}".format(N_MICE))
# print(len(LPATH_HIPPO_LFP_NPY_LIST_MICE))

# SDIR_CLEANLAB = "./data/okada/cleanlab_results/{}/".format(dataset_key)

################################################################################
## Loads
################################################################################
lfps, rips_df_list_GMM_labeled = utils.pj.load_lfps_rips_sec(
    LPATH_HIPPO_LFP_NPY_LIST_MICE, rip_sec_ver="GMM_labeled/{}".format(dataset_key)
)  # includes labels using GMM on the dataset
del lfps
lfps, rips_df_list_isolated = utils.pj.load_lfps_rips_sec(
    LPATH_HIPPO_LFP_NPY_LIST_MICE, rip_sec_ver="isolated"
)  # includes isolated LFP during each ripple candidate
del lfps


################################################################################
## Organizes rips_df
################################################################################
len_rips = [len(_rips_df_tt) for _rips_df_tt in rips_df_list_GMM_labeled]
rips_df_list_GMM_labeled = pd.concat(rips_df_list_GMM_labeled)
rips_df_list_isolated = pd.concat(rips_df_list_isolated)
rips_df = pd.concat([rips_df_list_GMM_labeled, rips_df_list_isolated], axis=1)  # concat
# Delete unnecessary columns
rips_df = rips_df.loc[:, ~rips_df.columns.duplicated()]  # delete duplicated columns
rips_df = rips_df[["start_sec", "end_sec", "are_ripple_GMM", "isolated"]]
# 'start_sec', 'end_sec',
# 'ln(duration_ms)', 'mean ln(MEP magni. / SD)', 'ln(ripple peak magni. / SD)',
# 'are_ripple_GMM'
