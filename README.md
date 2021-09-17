## Install ripple_detector_CNN
The paper "Towards threshold invariance in defining hippocampal ripples" introduced a hippocamal ripple detector. It is installed via the pip.
``` python
pip install ripple_detector_CNN
```
A demo page shows how to use the detector.

### Download mouse CA1 LFP data from the CRCNS repository
The th-1 dataset was used in the demo page. Please see dir .data/CRCN_batch_download/download/.


## Install analytical code
Please clone this repository and add to the PATH the top directory of this repository (.towards-threshold-invariance-in-defining-hippocampal-ripples). 

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
 '/mnt/md0/towards-threshold-invariance-in-defining-hippocampal-ripples/package/src',
 ...
 '.']
'''
```

### Build singularity container for executing the analytical code
``` bash
$ singularity build .singularity/towards_threshold_invariance_in_defining_hippocampal_ripples.sif .singularity/towards_threshold_invariance_in_defining_hippocampal_ripples.def
```

### Our machine specs
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
│   ├── singularity-aliases.bash (Bash aliases for using the singularity container)
│   ├── towards-threshold-invariance-in-defining-hippocampal-ripples.def (A singularity definition file)
├── ripples
│   ├── detect_ripples
│   ├── define_ripples
├── utils (other than pj has been transfered to the [mngs package](https://github.com/ywatanabe1989/mngs))
│   ├── dsp.py (**D**igital **S**ignal **P**rocessing)
│   ├── general.py (general code which are always written by pythonista)
│   ├── ml (**M**achine **L**earning)
│   ├── pj (unique for the **P**ro**j**ect)
│   ├── plt (**Pl**o**t**ting)
│   └── stats (**Stat**i**s**tics)
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
