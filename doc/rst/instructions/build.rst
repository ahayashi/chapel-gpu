=======================
Building Chapel-GPU
=======================

Prerequisites
##############

* Chapel: 1.22 or below. Detailed instructions for installing Chapel can be found: `here <https://chapel-lang.org/docs/usingchapel/QUICKSTART.html>`_.

* GPU Compilers & Runtimes: GPUIterator and GPUAPI require either of the following GPU programing environments.

   * NVIDIA CUDA: Tested with 10.2
   * AMD HIP: Tested with 2.8
   * OpenCL: Tested with 2.2 and 1.2

* Build tools

   * cmake: 3.8.0 or above

.. note:: While ``GPUIterator`` works with OpenCL, ``GPUAPI`` with OpenCL is under developement.

Instructions
##############

1. Clone the repository

.. code-block:: bash

   git clone https://github.com/ahayashi/chapel-gpu.git

2. Build ``libGPUAPI`` library using ``cmake``

.. code-block:: bash

   cd chapel-gpu
   mkdir build
   cd build
   cmake ..
   make
   make install

This produces the following files:

.. csv-table::
   :header: "File", "Type", "Destination", "Description"
   :widths: 20, 20, 20, 50

   env.sh, Shell Script, ``bin``, Sets environment variables.
   libGPUAPIX.so, Shared Library, ``lib``, "X is either CUDA, HIP, or OpenCL."
   libGPUAPIX_static.a, Static Library, ``lib``, "A static version of libGPUAPI. Mainly used in this document."
   GPUAPI.h, Header File, ``include``, "A header file that includes the declaration of GPUAPI functions."
   lambda.h, Header File, ``include``, "A header file that facilitates writing device lambdas."
   GPUIterator.chpl, Chapel Source File, ``modules``, "The ``GPUIterator`` module"
   GPUAPI.chpl, Chapel Source File, ``modules``, "The ``GPUAPI`` module"


By default, the libraries are installed into :code:`chapel-gpu/install`. If you wish to install it into your preferred directly, please type:

.. code-block:: bash

   cmake -DCMAKE_INSTALL_PREFIX=path/to/your_preferred_directory ..


.. note::
   **For CUDA Users**: If CUDA is not found, make sure :code:`nvcc` is in your path or tell :code:`cmake` the path to :code:`nvcc`. For example: :code:`cmake CMAKE_CUDA_COMPILER=path_to/nvcc ..`

   **For AMD HIP Users**: Chapel-GPU relies on :code:`hipify-perl` to convert CUDA programs to HIP programs internally. If you are pretty sure HIP is installed on your system, but :code:`cmake` complains :code:`hipify-perl` is not found, consider updating the following line in :code:`CMakeLists.txt`: :code:`if(EXISTS "${HIP_ROOT_DIR}/hip/bin/hipify-perl")`

3. source ``env.sh``

.. code-block:: bash

   cd ..
   source ./install/bin/env.sh

|
   This sets 1) ``$CHPL_GPU_HOME``, and 2) environment variables related to CUDA/HIP/OpenCL installation directory, the latter of which can be referred when the user creates object files for thier GPU programs.

4. Build and run a test program

  See :doc:`Compiling and running <compile>`
