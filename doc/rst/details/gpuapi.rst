.. default-domain:: chpl

===========
GPUAPI
===========

Overview
################

The GPUAPI module provides Chapel-level GPU API. The use of the API assumes cases where the user would like to 1) write GPU kernels in low-level GPU languages such as CUDA/HIP/OpenCL, or 2) utilize highly-tuned GPU libraries, and would like to stick with Chapel for the other parts (allocation, data transfers). Currently, it provides two tiers of GPU API:

* `MID-level`: Provides Chapel user-friendly GPU API functions.

  * Example: ``var ga = new GPUArray(A);``

* `MID-LOW-level`: Provides wrapper functions for raw GPU API functions

  * Example: ``var ga: c_void_ptr = Malloc(sizeInBytes);``



Further Readings
################

* Exploring a multi-resolution GPU programming model for Chapel. Akihiro Hayashi, Sri Raj Paul, Vivek Sarkar, 7th Annual Chapel Implementers and Users Workshop (CHIUW), May 2020. (co-located with IPDPS2020).

  .. youtube:: Mq_vhXlSHxU
