## Set envirionment variables
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
- [ ] Fig.01-02_Ripple candidates
- [ ] Fig.03_ Relationship between the hippocampal LFP and animal movement
- [ ] Fig.04_Ripple_candidates_EDA
- [ ] Fig.05_Gaussian_Mixture_Model_Clustering
- [ ] Fig.06-16_Defining_Ripples_using_Confident_Learning
- [ ] Fig.17-20_Detecting_Ripples_usign_a_CNN
- [ ] Fig.21_Estimating_Optimal_Threshold_for_the_Ripple_Peak_Magnitude

