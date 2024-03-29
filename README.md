## Ripple Detector
A hippocamal ripple detector, introduced in the paper "Towards threshold invariance in defining hippocampal ripples", can be installed via the pip.
``` bash
$ pip install ripple_detector_CNN
```
It's also included in [the prepaired singularity container](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/singularity/towards.def).


## Trained Weights
Please install the git-lfs for downloading [the trained weights](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/ripples/detect_ripples/CNN/train_FvsT/checkpoints/). Installation scripts are written on [the official page](https://packagecloud.io/github/git-lfs/install).
``` bash
$ git lfs install
```


## Available LFP Data and a Demo
To reproduce [a demo](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/ripples/detect_ripples/CNN/from_unseen_LFP.ipynb), please download [the th-1 dataset](https://crcns.org/data-sets/thalamus/th-1/about-th-1) of CRCNS. The demo (.ipynb/.py) shows how to use the ripple_detector_CNN with unseen LFP data.

>Peyrache, A., Petersen P., Buzsáki, G. (2015)  
>Extracellular recordings from multi-site silicon probes in the anterior thalamus and subicular formation of freely moving mice. CRCNS.org.  
>http://dx.doi.org/10.6080/K0G15XS1  

Scripts for downloading the th-1 dataset is prepaired under [./data/CRCN_batch_download/download/](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/tree/main/data/CRCN_batch_download/download), using the official downloader. To reproduce the demo, please set the following two files as below.
``` bash
./towards-threshold-invariance-in-defining-hippocampal-ripples/data/th-1/data/Mouse12-120806/Mouse12-120806.eeg
./towards-threshold-invariance-in-defining-hippocampal-ripples/data/th-1/data/Mouse12-120806/Mouse12-120806.xml
```

The .py version of the demo is also run with the command below and the results will be logged under [./ripples/detect_ripples/CNN/from_unseen_LFP/](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/ripples/detect_ripples/CNN/from_unseen_LFP/).

``` bash
$ singularity exec --nv ./singularity/image.sif python ./ripples/detect_ripples/CNN/from_unseen_LFP.py
# $ spy ./ripples/detect_ripples/CNN/from_unseen_LFP.py # using ./singularity/singularity-aliases.bash
```



## Analytical Code
Please clone this repository and add the top directory to the PATH.
``` bash
$ git clone https:github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples
$ cd towards-threshold-invariance-in-defining-hippocampal-ripples
$ du -sh ripples/detect_ripples/CNN/train_FvsT/checkpoints/ # 612MB if git-lfs was installed otherwise 20KB
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


## The Singularity Container for the Analytical Code
``` bash
$ singularity build .singularity/towards.sif .singularity/towards.def
$ singularity exec --nv ./singularity/image.sif python ./ripples/detect_ripples/CNN/from_unseen_LFP.py # an example
# $ spy ./ripples/detect_ripples/CNN/from_unseen_LFP.py # using ./singularity/singularity-aliases.bash
```


## Directory Tree
```
.towards-threshold-invariance-in-defining-hippocampal-ripples
├── conf (global configuration files)
├── fig_making_scripts (softlinks to scripts to make figures for a paper)
├── models (our CNN definition files (.py) and configuration files (.yaml))
├── README.md (this file)
├── singularity 
│   └── towards.def (a singularity definition file)
├── ripples
│   ├── detect_ripples
│   └── define_ripples
├── utils (Now, utils is transfered to [mngs package](https://github.com/ywatanabe1989/mngs) except for the pj dir)
│   ├── dsp.py (**d**igital **s**ignal **p**rocessing)
│   ├── general.py (general snippets for python users)
│   ├── ml (**m**achine **l**earning)
│   ├── pj (unique for this **p**ro**j**ect)
│   ├── plt (**pl**o**t**ting)
│   └── stats (**stat**i**s**tics)
└── paper (info for this paper)
```


## The Order for the Analysis
- signal conversion to python numpy format
    - [./data/okada/preprocessing/nsx2mat_octave.m](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/data/okada/preprocessing/nsx2mat_octave.m)
    - [./data/okada/preprocessing/mat2npy.py](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/data/okada/preprocessing/mat2npy.py)
    - [./data/okada/preprocessing/mouse05_48h_to_2days.py](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/data/okada/preprocessing/mouse05_48h_to_2days.py)
    - [./data/okada/preprocessing/mk_fpaths_list_of_hippo_LFP_and_trape_MEP_tets.sh](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/data/okada/preprocessing/mk_fpaths_list_of_hippo_LFP_and_trape_MEP_tets.sh)

- defininig ripple candidates and calculates the properties
  - [./ripples/define_ripples/conventional/sh_scripts/makes_labels.sh](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/ripples/define_ripples/conventional/sh_scripts/makes_labels.sh)
  - [./ripples/define_ripples/conventional/sh_scripts/extracts_bands_magnitude.sh](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/ripples/define_ripples/conventional/sh_scripts/extracts_bands_magnitude.sh)
  - [./ripples/define_ripples/conventional/sh_scripts/calc_props.sh](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/ripples/define_ripples/conventional/sh_scripts/calc_props.sh)
  - [./ripples/define_ripples/conventional/sh_scripts/plots_prop_hists.sh](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/ripples/define_ripples/conventional/sh_scripts/plots_prop_hists.sh)
  - [./ripples/define_ripples/conventional/sh_scripts/plots_3d_scatter.sh](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/ripples/define_ripples/conventional/sh_scripts/plots_3d_scatter.sh)
  
- checking the correlation of MEP of trapeziuses and FFT powers at frequencies 0-499 Hz
  - [./EDA/sh_scripts/MEP_FFT_pow_corr.sh](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/EDA/sh_scripts/MEP_FFT_pow_corr.sh)

- defining ripples using a [GMM (Gaussian mixture model)](https://scikit-learn.org/stable/modules/mixture.html) clustering
  - [./ripples/define_ripples/using_GMM/estimates_the_optimal_n_clusters.py](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/ripples/define_ripples/using_GMM/estimates_the_optimal_n_clusters.py)
  - [./ripples/define_ripples/using_GMM/sh_scripts/makes_labels.sh](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/ripples/define_ripples/using_GMM/sh_scripts/makes_labels.sh)
  - [./ripples/define_ripples/using_GMM/sh_scripts/makes_labels_D0X-.sh](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/ripples/define_ripples/using_GMM/sh_scripts/makes_labels_D0X-.sh)
  - [./ripples/define_ripples/using_GMM/sh_scripts/plots_3d_scatter.sh](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/ripples/define_ripples/using_GMM/sh_scripts/plots_3d_scatter.sh)

- defining ripples using [a designed CNN model; ResNet1D](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/models/ResNet1D/ResNet1D.py)
  - [./ripples/define_ripples/using_CNN/sh_scripts/isolates_candidates.sh](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/ripples/define_ripples/using_CNN/sh_scripts/isolates_candidates.sh)
  - [./ripples/define_ripples/using_CNN/sh_scripts/makes_labels.sh](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/ripples/define_ripples/using_CNN/sh_scripts/makes_labels.sh)
  - [./ripples/define_ripples/using_CNN/sh_scripts/plots_3d_scatter.sh](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/ripples/define_ripples/using_CNN/sh_scripts/plots_3d_scatter.sh)
  - [./ripples/define_ripples/using_CNN/sh_scripts/checks_traces.sh](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/ripples/define_ripples/using_CNN/sh_scripts/checks_traces.sh)
  - [./ripples/define_ripples/summary/sh_scripts/checks_ripple_props.sh](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/ripples/define_ripples/summary/sh_scripts/checks_ripple_props.sh)
  - [./ripples/define_ripples/using_CNN/calcs_corr_of_labels.py](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/ripples/define_ripples/using_CNN/calcs_corr_of_labels.py)

- detecting ripples using [the trained CNN model](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/tree/main/ripples/detect_ripples/CNN/train_FvsT)
  - [./ripples/detect_ripples/CNN/train_FvsT.py](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/ripples/detect_ripples/CNN/train_FvsT.py)
  - [./ripples/detect_ripples/CNN/from_unseen_LFP.py](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/ripples/detect_ripples/CNN/from_unseen_LFP.py)
  - [./ripples/detect_ripples/CNN/from_unseen_LFP.ipynb (the demo page)](https://github.com/ywatanabe1989/towards-threshold-invariance-in-defining-hippocampal-ripples/blob/main/ripples/detect_ripples/CNN/from_unseen_LFP.ipynb)

    
## Machine Info
### For just estimating ripple probabilities using the once trained model
    - CPU: AMD Ryzen 7 1700 Eight-Core Processor
    - RAM: 64 GB
    - GPU: NVIDIA GeForce GTX 1070
    - 1 TB NVME storage
    - Nvidia Driver: 470.63.01
    - CUDA version: V10.2.89

### For training the model from scratch 
    - CPU: Intel(R) Core(TM) i7-6950X CPU @ 3.00GHz
    - RAM: 128 GB
    - GPU: NVIDIA GeForce GTX 1080 Ti * 4
    - 2 TB NVME storage
    - (28 TB RAID5 HDD storage)
    - Nvidia Driver: 465.19.01
    - CUDA version: V10.1.243

## Contact
