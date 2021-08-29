## Installation
To run the code, please clone this repository and add to the PATH this repository's top directory (.towards-threshold-invariance-in-defining-hippocampal-ripples). To check if it is correctly recognized from your python environment, executing the following python code would be helpful.

``` python
import sys
print(sys.path)
'''
['/usr/local/bin',
 ...
 '/home/<USERNAME>/.ipython',
 '.']
'''
```

## Building .sif file (Singularity Image File)

``` bash
$ singularity build .singularity/towards_threshold_invariance_in_defining_hippocampal_ripples.sif
```

## Our machine specs
- CPU: Intel(R) Core(TM) i7-6950X CPU @ 3.00GHz
- RAM: 128 GB
- GPU: NVIDIA GeForce GTX 1080 Ti * 4
- 2 TB NVME storage
- (28 TB RAID5 HDD storage)
- Nvidia Driver: 465.19.01
- CUDA version: V10.1.243


## File/directory description (only major ones)
```
.towards-threshold-invariance-in-defining-hippocampal-ripples
├── conf (Globally used configuration files)
├── fig_making_scripts (Softlinks to scripts to make figures for a paper)
├── models (Definition files (.py) and configuration files (.yaml) for deep learning models and modules for them)
├── README.md (This file.)
├── singularity 
│   ├── singularity.bash (Bash aliases for using the singularity container)
├── ripples
│   ├── detect_ripples
│   ├── define_ripples
├── utils
│   ├── dsp.py ("D"igital "S"ignal "P"rocessing)
│   ├── general.py (general code which are always written by pythonista)
│   ├── ml ("M"achine "L"earning)
│   ├── pj (unique for the "P"ro"j"ect)
│   ├── plt ("Pl"o"t"ting)
│   └── stats ("Stat"i"s"tics)
├── paper (info for this paper)
```


## The order scripts were executed.
./data/okada/preprocessing/nsx2mat_octave.m
./data/okada/preprocessing/mat2npy.py
./data/okada/preprocessing/mouse05_48h_to_2days.py
./data/okada/preprocessing/mk_fpaths_list_of_hippo_LFP_and_trape_MEP_tets.sh

./ripples/define_ripples/conventional/sh_scripts/makes_labels.sh
./ripples/define_ripples/conventional/sh_scripts/extracts_bands_magnitude.sh
./ripples/define_ripples/conventional/sh_scripts/calc_props.sh

./EDA/sh_scripts/MEP_FFT_pow_corr.sh

./ripples/define_ripples/conventional/sh_scripts/plots_prop_hists.sh
./ripples/define_ripples/using_GMM/estimates_the_optimal_n_clusters.py

./ripples/define_ripples/conventional/sh_scripts/plots_3d_scatter.sh

./ripples/define_ripples/using_GMM/sh_scripts/makes_labels.sh
./ripples/define_ripples/using_GMM/sh_scripts/makes_labels_D0X-.sh
./ripples/define_ripples/using_GMM/sh_scripts/plots_3d_scatter.sh

./ripples/define_ripples/using_CNN/sh_scripts/isolates_candidates.sh
./ripples/define_ripples/using_CNN/sh_scripts/makes_labels.sh

./ripples/define_ripples/using_CNN/plots_3d_scatter.py
./ripples/define_ripples/using_CNN/sh_scripts/plots_3d_scatter.sh
./ripples/define_ripples/using_CNN/sh_scripts/checks_traces.sh

./ripples/define_ripples/summary/sh_scripts/checks_ripple_props.sh
./ripples/define_ripples/using_CNN/calcs_corr_of_labels.py


./ripples/detect_ripples/CNN/train_FvsT.py

################################################################################
./ripples/detect_ripples/CNN/from_unseen_LFP.py


## TODO
- [ ] Experiment on hc-19
    - [x] downloads hc-19
    
- [ ] to open source models and weights with pip

./ripples/detect_ripples/CNN/train_FvsT/checkpoints/mouse_test#01_epoch#000.pth
