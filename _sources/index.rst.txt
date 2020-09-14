.. role:: chapel(code)
   :language: chapel

Chapel GPU Documentation
================================

Overview
--------
This document describes the following two Chapel modules that facilitate GPU programming in Chapel:

* `GPUIterator`: A Chapel iterator that facilitates invoking user-written GPU programs (e.g., CUDA/HIP/OpenCL) from Chapel programs. It is also designed to easily perform hybrid and/or distributed execution - i.e., CPU-only, GPU-only, X% for CPU + Y% for GPU on a single or multiple CPU+GPU node(s), which helps the user to explore the best configuration.

* `GPUAPI`: Chapel-level GPU API that allows the user to perform basic operations such as GPU memory (de)allocations, device-to-host/host-to-device transfers, and so on. This module can be used either standalone or with the GPUIterator module. Currently, the following two tiers of API are provided:

    * `MID-level`: Provides Chapel user-friendly GPU API functions.

       * Example: :chapel:`var ga = new GPUArray(A);`

    * `LOW-MID-level`: Provides wrapper functions for raw GPU API functions

       * Example: :chapel:`var ga: c_void_ptr = Malloc(sizeInBytes);`


Also, in this document, for categorization purposes, the term `LOW-level` is referred to a GPUIterator only version, where the GPUIterator is only used for invoking raw GPU programs in which there is no Chapel-level abstraction.


.. toctree::
   :maxdepth: 2
   :caption: QuickStart Instructions

   instructions/build
   instructions/write

.. toctree::
   :maxdepth: 2
   :caption: Technical Details

   details/gpuiterator
   details/gpuapi

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/gpuiterator
   api/gpuapi

.. toctree::
   :maxdepth: 2
   :caption: History

   history/evolution
