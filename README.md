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

./ripples/define_ripples/conventional/define_ripple_candidates.py
./ripples/define_ripples/conventional/extracts_bands_magnitude.py

./EDA/MEP_FFT_pow_corr.py

./ripples/define_ripples/conventional/plots_props_hists.py
./ripples/define_ripples/conventional/plots_3d_scatter.py
  
./ripples/define_ripples/using_GMM/makes_labels.py
./ripples/define_ripples/using_GMM/plots_3d_scatter.py

./ripples/define_ripples/using_CNN/isolates_candidates.py
./ripples/define_ripples/using_CNN/makes_labels.py
./ripples/define_ripples/using_CNN/plots_3d_scatter.py


## ./data dir tree 
$ tree ./data > ./data/data_tree.txt


## To-do
- [x] Fig.01
- [x] Fig.02
- [x] Fig.03
- [x] Fig.04
- [x] Fig.05
- [x] Fig.06
- [x] Fig.07
- [x] Fig.08
- [x] Fig.09
- [x] Fig.10
- [x] Fig.11
- [ ] Fig.12
- [ ] Fig.13
- [ ] Fig.14
- [ ] Fig.15
- [ ] Fig.16
- [x] Fig.17
- [ ] Fig.18
- [ ] Fig.19
- [ ] Fig.20
- [ ] Fig.21

- [x] Estimates the optimal number of GMM clusters
- [ ] Enables ripple_detection module to enjoy GPU acceleration


- [ ] checks if GMM labels were wrongly saved
- [ ] transfer learning on hc-22/25 (CRCNS.org)


## fixes GMM labels are inversed
- [ ] ./ripples/define_ripples/using_GMM/makes_labels.py
  - [x] script itself
  - [ ] 
- [ ] ./ripples/define_ripples/using_GMM/plots_3d_scatter.py

- [ ] ./ripples/define_ripples/using_CNN/isolates_candidates.py
- [ ] ./ripples/define_ripples/using_CNN/makes_labels.py
- [ ] ./ripples/define_ripples/using_CNN/plots_3d_scatter.py
