## Installation
To run the code, please download or clone this repository and add this repositories top directory (./DL_REPO/) to the PYTHONPATH. To check which paths are recognized from your python environment, executing the following 2-line python code would be the easiest.

``` python
import sys
print(sys.path)
'''
['/usr/local/bin',
 '/usr/lib64/python38.zip',
 '/usr/lib64/python3.8',
 '/usr/lib64/python3.8/lib-dynload',
 '',
 '/home/ywatanabe/.local/lib/python3.8/site-packages',
 '/usr/local/lib64/python3.8/site-packages',
 '/usr/local/lib/python3.8/site-packages',
 '/usr/lib64/python3.8/site-packages',
 '/usr/lib/python3.8/site-packages',
 '/usr/local/lib/python3.8/site-packages/IPython/extensions',
 '/home/ywatanabe/.ipython',
 '.',
 '.']
'''
```

## Our machine specs
- CPU: Intel(R) Core(TM) i7-6950X CPU @ 3.00GHz
- RAM: 128 GB
- GPU: NVIDIA GeForce GTX 1080 Ti * 4
- 2 TB NVME storage
- (28 TB RAID5 HDD storage)
- Nvidia Driver: 465.19.01
- CUDA version: V10.1.243



``` python
## Sets tee
sys.stdout, sys.stderr = utils.general.tee(sys)
```


## File/directory description (only major ones)
```
.DL_REPO
├── conf (Globally used configuration files)
├── fig_making_scripts (Softlinks to scripts to make figures for a paper)
├── models (Definition files (.py) and configuration files (.yaml) for deep learning models and modules for them)
├── README.md (This file.)
├── singularity 
│   ├── singularity.bash (Bash aliases for using singularity as lasy as possible for each project)
├── ripples
│   ├── detect_ripples
│   ├── define_ripples
├── **utils (Ideally, this directory "utils" would be transfererable to any project other than ./utils/pj/ ("p"ro"j"ect). However, there remain some dependancy problems.)**
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
./ripples/define_ripples/conventional/sh_scripts/plots_3d_scatter.sh

./ripples/define_ripples/using_GMM/sh_scripts/estimates_the_optimal_n_clusters.sh
./ripples/define_ripples/using_GMM/sh_scripts/makes_labels.sh
./ripples/define_ripples/using_GMM/sh_scripts/plots_3d_scatter.sh

./ripples/define_ripples/using_CNN/sh_scripts/isolates_candidates.sh
./ripples/define_ripples/using_CNN/sh_scripts/makes_labels.sh



### From here!!! ###
./ripples/define_ripples/using_CNN/plots_3d_scatter.py
./ripples/define_ripples/using_CNN/sh_scripts/plots_3d_scatter.sh


./ripples/define_ripples/using_CNN/sh_scripts/checks_traces.sh # fixed

./ripples/define_ripples/summary/checks_avg_traces.py
./ripples/define_ripples/summary/checks_ripple_props.py

./ripples/detect_ripples/CNN/train_n_vs_r.py
./ripples/detect_ripples/CNN/sh_scripts/predict_n_s_r.sh
./ripples/detect_ripples/CNN/sh_scripts/fit_sigmoid_on_the_predicted_scores_of_s.sh



  


## ./data dir tree 
$ tree ./data > ./data/data_tree.txt

## To-Dos
- [x] Fix Confident Learning
    - [x] D01-
    - [x] D02-
    - [x] D03
    - [x] D04-
    - [x] D05-

- [x] Fig.12: Represent traces for defining ripples using CNN (confident learning)
    - [x] fixed
    
- [ ] Fig. 13: Ripple props

- [ ] Experiment on hc-19
    - [x] downloads hc-19
    
- [ ] to fix ripples/define_ripples/using_CNN/checks_traces.py
- [ ] to open source models and weights with pip

