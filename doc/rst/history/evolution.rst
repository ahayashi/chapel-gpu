=============================================
Chapel-GPU Evolution
=============================================

version 0.4.0, October, 2021
############################

Version 0.4.0 adds the following new features to version 0.3.0:

- Update the GPUAPI module to support Intel DPC++ (SYCL).
- Update the installation script to install the package configuration files ``gpuapi[cuda|hip|dpcpp]-config.cmake`` in ``$(CHPL_GPU_HOME)/share/cmake/ChapelGPUAPI``, which allows the user to utilize the GPUAPI library from their cmake project.

version 0.3.0, June, 2021
############################

Version 0.3.0 adds the following new features to version 0.2.0:

- Update the GPUAPI module to add 1) pitched 2D array support at MID-LOW-level, and 2) jagged array suuport at the MID-level.

version 0.2.0, August, 2020
############################

Version 0.2.0 adds the following new features to version 0.1.0:

- Add the ``GPUAPI`` module, which provides Chapel-level GPU API and reduces the complexity of writing data (de)allocations, and transfers for GPU execution.
- Introduce a ``cmake``-based build system, which facilitates building GPU prorams on different GPU platforms (CUDA, HIP, and OpenCL).

version 0.1.0, July, 2019
###########################

Version 0.1.0 provides an initial version of the ``GPUIterator`` module, which facilitates the invocation of user-provided GPU programs from Chapel programs. 
