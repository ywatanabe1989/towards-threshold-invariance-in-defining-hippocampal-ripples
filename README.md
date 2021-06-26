## Sets envirionment variables
``` bash
$ export SEMI_RIPPLE_HOME=/mnt/nvme/Semisupervised_Ripple
$ export SINGULARITY_SIF_ROOT_NAME=semi_ripples
$ export SINGULARITY_BINDPATH="/mnt,"
$ . $SEMI_RIPPLE_HOME/singularity/singularity-aliases.bash # aliases
$ export SINGULARITY_SHELL=/bin/bash
$ # export PYTHONPATH=$SEMI_RIPPLE_HOME:$PYTHONPATH
```

## Our machine specs
- CPU: Intel(R) Core(TM) i7-6950X CPU @ 3.00GHz
- RAM: 128 GB
- GPU: NVIDIA GeForce GTX 1080 Ti * 4
- 2 TB NVME storage
- (28 TB RAID5 HDD storage)
- Nvidia Driver: 465.19.01
- CUDA version: V10.1.243


## The order scripts were executed.
./data/okada/preprocessing/nsx2mat_octave.m
./data/okada/preprocessing/mat2npy.py
./data/okada/preprocessing/mouse05_48h_to_2days.py
./data/okada/preprocessing/mk_fpaths_list_of_hippo_LFP_and_trape_MEP_tets.sh


./ripples/define_ripples/conventional/sh_scripts/make_labels.sh
./ripples/define_ripples/conventional/sh_scripts/extracts_bands_magnitude.sh
./ripples/define_ripples/conventional/sh_scripts/calc_props.sh


./EDA/sh_scripts/MEP_FFT_pow_corr.sh


./ripples/define_ripples/conventional/sh_scripts/plots_prop_hists.sh
./ripples/define_ripples/conventional/sh_scripts/plots_3d_scatter.sh


./ripples/define_ripples/using_GMM/sh_scripts/estimates_the_optimal_n_clusters.sh
./ripples/define_ripples/using_GMM/sh_scripts/makes_labels.sh
./ripples/define_ripples/using_GMM/sh_scripts/plots_3d_scatter.sh


./ripples/define_ripples/using_CNN/sh_scripts/isolates_candidates.sh
./ripples/define_ripples/using_CNN/sh_scripts/makes_labels.sh
./ripples/define_ripples/using_CNN/sh_scripts/plots_3d_scatter.sh
./ripples/define_ripples/using_CNN/checks_traces.py
./ripples/define_ripples/using_CNN/checks_avg_traces.py

./ripples/detect_ripples/CNN/train_n_vs_r.py
./ripples/detect_ripples/CNN/sh_scripts/predict_n_s_r.sh
./ripples/detect_ripples/CNN/sh_scripts/fit_sigmoid_on_the_predicted_scores_of_s.sh



## ./data dir tree 
$ tree ./data > ./data/data_tree.txt


## To-do
- [ ] Fig.18
- [ ] Fig.19
- [ ] Fig.20
- [ ] Fig.21


- [ ] Enables ripple_detection module to enjoy GPU acceleration

- [ ] transfer learning on hc-22/25 (CRCNS.org)
  - [x] downloads hc-22
  - [x] downloads hc-25  


- [ ] organize utils

- [ ] makes labels with existing methods
