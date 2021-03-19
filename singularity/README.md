## ./host
"./host/" contains shell scripts to enable NVIDIA's GPU support and install singularity environment on a host machine. We installed NVIDIA Driver version 460.32.03, CUDA version 10.1.243, and singularity 3.6.3 on a host machine. For more details, please see the official documentations.
- [NVIDIA Driver Installation Quickstart](https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_Driver_Installation_Quickstart.pdf)
- [CUDA Toolkit Documentation v10.1.243](https://docs.nvidia.com/cuda/archive/10.1/)
- [Singularity GPU Support](https://sylabs.io/guides/3.6/user-guide/gpu.html)

## ./semi_ripples.def
"./semi_ripples.def" is the singularity definition file for "semi_ripples.sif."

## ./semi_ripples.sif
"./semi_ripples.sif" is the singularity image file to run our python scripts.

