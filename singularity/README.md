## ./host
**./host/** contains shell scripts to enable NVIDIA's GPU support and install singularity environment on a host machine. We installed NVIDIA Driver v460.32.03, CUDA version v10.1.243, and singularity v3.6.3 on a host machine. For more details, please see the official documentations.
- [NVIDIA Driver Installation Quickstart](https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_Driver_Installation_Quickstart.pdf)
- [CUDA Toolkit Documentation v10.1.243](https://docs.nvidia.com/cuda/archive/10.1/)
- [Singularity GPU Support](https://sylabs.io/guides/3.6/user-guide/gpu.html)

## ./towards_threshold_invariance_in_defining_hippocampal_ripples.def
**./towards_threshold_invariance_in_defining_hippocampal_ripples.def** is the singularity definition file for **towards_threshold_invariance_in_defining_hippocampal_ripples.sif**.

## ./towards_threshold_invariance_in_defining_hippocampal_ripples.sif
**./towards_threshold_invariance_in_defining_hippocampal_ripples.sif** is the singularity image file to run our python scripts.

## ./singularity-aliases.bash
**./singularity-aliases.bash** contains bash aliases to use singularity casually.

``` bash
$ sshell # move into the shell in the singularity environment (**s**ingularity **shell**).
$ sipy # starts ipython in the singularity environment (**s**ingularity **ipy**ton).
$ spy *.py # executs a python program in the singularity environment (**s**ingularity **py**thon).
$ sbuild *.def # build a singularity Definition file. The second arguments takes --fakeroot (-f) or --remote (-r) to build to *sif file (e.g., $ sbuild <FILENAME>.def --fakeroot).

$ sshellw # writable sshell
$ sipyw # writable sipy
$ spyw *.py # writable spy
$ sbuildw # writable sbuild
```


