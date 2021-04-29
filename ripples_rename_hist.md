ripple_candi_1kHz_pkl/
├── orig
├── with_GMM_preds
└── with_props



ripples_1kHz_csv/
├── candi_orig
├── GMM_labeled
└── candi_with_props
(├── CNN_labeled)


rename ripple_candi_1kHz_pkl  ripples_1kHz_csv  0?/day?/split/ripple_candi_1kHz_pkl
rename orig                   candi_orig        0?/day?/split/ripples_1kHz_csv/orig
rename with_GMM_preds         GMM_labeled       0?/day?/split/ripples_1kHz_csv/with_GMM_preds
rename with_props             candi_with_props  0?/day?/split/ripples_1kHz_csv/with_props



################################################################################
csv to pkl
################################################################################
rename ripples_1kHz_csv  ripples_1kHz_pkl  0?/day?/split/ripples_1kHz_csv


ls 0?/day?/split/ripples_1kHz_pkl

ripples_1kHz_csv/
├── candi_orig
├── GMM_labeled
└── candi_with_props
(├── CNN_labeled)
