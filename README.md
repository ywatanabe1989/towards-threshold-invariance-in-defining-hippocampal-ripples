## Sget envirionment variables
``` bash
export SEMI_RIPPLE_HOME=/mnt/nvme/Semisupervised_Ripple
export SINGULARITY_SIF_ROOT_NAME=semi_ripples
export SINGULARITY_BINDPATH="/mnt,"
. $SEMI_RIPPLE_HOME/singularity/singularity-aliases.bash # aliases
export SINGULARITY_SHELL=/bin/bash
```

export PYTHONPATH=/mnt/nvme/Semisupervised_Ripple


## Machine specs
- CPU:
- RAM: 128 GB
- GPU: 1080ti * 4
- 2 TB NVME storage
- 28 TB RAID5 HDD storage



## To-do
- [x] Fig.01
- [x] Fig.02
- [x] Fig.03
- [x] Fig.04
- [x] Fig.05
- [x] Fig.06
- [ ] Fig.07
- [x] Fig.08
- [ ] Fig.09
- [ ] Fig.10
- [ ] Fig.11
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



- [ ] Fig.01-02_Ripple candidates
- [ ] Fig.03_ Relationship between the hippocampal LFP and animal movement
- [ ] Fig.04_Ripple_candidates_EDA
- [ ] Fig.05_Gaussian_Mixture_Model_Clustering
- [ ] Fig.06-16_Defining_Ripples_using_Confident_Learning
- [ ] Fig.17-20_Detecting_Ripples_usign_a_CNN
- [ ] Fig.21_Estimating_Optimal_Threshold_for_the_Ripple_Peak_Magnitude


## sh_scripts template
cat $SEMI_RIPPLE_HOME/data/okada/FPATH_LISTS/HIPPO_LFP_TT_NPYs.txt |
xargs -P $N_CPUS -n 1 python3 ./ripples/define_ripples/candidates/detects_ripple_candidates.py -n 2>&1 | tee $0.log
