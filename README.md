## Install ripple_detector_CNN
A hippocamal ripple detector, introduced in the paper "Towards threshold invariance in defining hippocampal ripples", can be installed via the pip.
``` bash
$ pip install ripple_detector_CNN
```
## Install git-lfs for the trained weights
https://packagecloud.io/github/git-lfs/install

``` bash
$ git lfs install
```


### CRCNS's opensourced-data
[A demo page](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/ripples/detect_ripples/CNN/from_unseen_LFP.ipynb) shows how to use the ripple_detector_CNN with LFP data from a mouse's hippocampal CA1 region in [the th-1 dataset](https://crcns.org/data-sets/thalamus/th-1/about-th-1). 

>Peyrache, A., Petersen P., Buzsáki, G. (2015)  
>Extracellular recordings from multi-site silicon probes in the anterior thalamus and subicular formation of freely moving mice. CRCNS.org.  
>http://dx.doi.org/10.6080/K0G15XS1  

Scripts for downloading the th-1 dataset is prepaired under [./data/CRCN_batch_download/download/](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/tree/main/data/CRCN_batch_download/download), using the official downloader. To reproduce the demo, please set the following two files as below.
``` bash
./towards-threshold-invariance-in-defining-hippocampal-ripples/data/th-1/data/Mouse12-120806/Mouse12-120806.eeg
./towards-threshold-invariance-in-defining-hippocampal-ripples/data/th-1/data/Mouse12-120806/Mouse12-120806.xml
```


## Install analytical code
Please clone this repository and add the top directory to the PATH.
``` bash
$ git clone https:github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples
$ cd towards-threshold-invariance-in-defining-hippocampal-ripples
$ git submodule init
$ git submodule update
```

The PATH can be checked like below.
``` python
import sys
print(sys.path)
'''
['/usr/local/bin',
 '/mnt/md0/towards-threshold-invariance-in-defining-hippocampal-ripples',
 ...
 '.']
'''
```


### Build singularity container for executing the analytical code
``` bash
$ singularity build .singularity/towards_threshold_invariance_in_defining_hippocampal_ripples.sif .singularity/towards_threshold_invariance_in_defining_hippocampal_ripples.def
```


## Directory tree with notes (only important ones)
```
.towards-threshold-invariance-in-defining-hippocampal-ripples
├── conf (global configuration files)
├── fig_making_scripts (softlinks to scripts to make figures for a paper)
├── models (our CNN definition files (.py) and configuration files (.yaml))
├── README.md (this file)
├── singularity 
│   ├── towards-threshold-invariance-in-defining-hippocampal-ripples.def (a singularity definition file)
├── ripples
│   ├── detect_ripples
│   ├── define_ripples
├── utils (Now, utils is transfered to [mngs package](https://github.com/ywatanabe1989/mngs) except for the pj dir)
│   ├── dsp.py (**d**igital **s**ignal **p**rocessing)
│   ├── general.py (general snippets for python users)
│   ├── ml (**m**achine **l**earning)
│   ├── pj (unique for this **p**ro**j**ect)
│   ├── plt (**pl**o**t**ting)
│   └── stats (**stat**i**s**tics)
├── paper (info for this paper)
```


## The order for the analytical scripts
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

./ripples/detect_ripples/CNN/from_unseen_LFP.py
./ripples/detect_ripples/CNN/from_unseen_LFP.ipynb


### Our machine info
- CPU: Intel(R) Core(TM) i7-6950X CPU @ 3.00GHz
- RAM: 128 GB
- GPU: NVIDIA GeForce GTX 1080 Ti * 4
- 2 TB NVME storage
- (28 TB RAID5 HDD storage)
- Nvidia Driver: 465.19.01
- CUDA version: V10.1.243
