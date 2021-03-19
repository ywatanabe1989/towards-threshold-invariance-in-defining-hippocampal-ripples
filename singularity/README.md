## [Singularity GPU Support](https://sylabs.io/guides/3.7/user-guide/gpu.html)
The host machine needs to be installed a driver and library for CUDA/ROCm.
```
As long as the host has a driver and library installation for CUDA/ROCm then it’s possible to e.g. run tensorflow in an up-to-date Ubuntu 18.04 container, from an older RHEL 6 host.
```

## Notes
- Ensure that **the /dev/nvidiaX** device entries are available inside the container, so that the GPU cards in the host are accessible.
- **Locate and bind the basic CUDA libraries from the host into the container**, so that they are available to the container, and **match the kernel GPU driver on the host**.
- Set the **LD_LIBRARY_PATH** inside the container so that **the bound-in version of the CUDA libraries** are used by applications run inside the container.

## Requirements
- The host has a working installation of the NVIDIA GPU driver, and a matching version of the basic NVIDIA/CUDA libraries. The host does not need to have an X server running, unless you want to run graphical apps from the container.
- Either a working installation of **the nvidia-container-cli tool** is available on the PATH when you run singularity, or the NVIDIA libraries are **in the system’s library search path**.
- The application inside your container was compiled for a CUDA version, and device capability level, that is supported by the host card and driver.

## Library Search Options
Singularity will find the NVIDIA/CUDA libraries on your host either using **the nvidia-container-cli tool**, or, if it is not available, a list of libraries in the configuration file etc/singularity/nvbliblist.

**If possible we recommend installing the nvidia-container-cli tool** from [the NVIDIA libnvidia-container website](https://nvidia.github.io/libnvidia-container/)
