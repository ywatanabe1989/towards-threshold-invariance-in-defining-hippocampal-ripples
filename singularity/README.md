## ./host
**./host/** contains shell scripts to enable NVIDIA's GPU support and install singularity environment on a host machine. We installed NVIDIA Driver v460.32.03, CUDA version v10.1.243, and singularity v3.6.3 on a host machine. For more details, please see the official documentations.
- [NVIDIA Driver Installation Quickstart](https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_Driver_Installation_Quickstart.pdf)
- [CUDA Toolkit Documentation v10.1.243](https://docs.nvidia.com/cuda/archive/10.1/)
- [Singularity GPU Support](https://sylabs.io/guides/3.6/user-guide/gpu.html)

## ./towards.def
The singularity definition file for **towards.sif**.

## ./towards.sif
The singularity image file to run our python scripts.

## ./singularity-aliases.bash
It contains bash aliases to use singularity easily.

``` bash
$ sshell # enter the shell of the singularity container (**s**ingularity **shell**).
$ sipy # run ipython from the singularity container (**s**ingularity **ipy**ton).
$ spy *.py # run python from the singularity container (**s**ingularity **py**thon).
$ sbuild *.def # build a singularity definition file. 
$ sbuild *.def -f # --fakeroot building
$ sbuild *.def -r # --remote building

$ sshellw # writable sshell
$ sipyw # writable sipy
$ spyw *.py # writable spy
$ sbuildw # writable sbuild
```


