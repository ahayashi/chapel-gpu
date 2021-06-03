=============================================
Compiling and Running Applications
=============================================

The repository has several example applications in ``chapel-gpu/example`` and ``chapel-gpu/apps`` directory, most of which has a distributed version:

.. csv-table::
   :header: "Benchmark", "Location", "Description", "Note"
   :widths: 20, 20, 20, 20

   Vector Copy, ``example`` and ``apps/vector_copy``, A simple vector kernel,
   STREAM, ``apps/stream``, `A = B + alpha * C`,
   BlackScholes, ``apps/blackscholes``, The Black-Scholes Equation,
   Logistic Regression, ``apps/logisticregression``, A classification algorithm,
   Matrix Multiplication, ``apps/mm``, Matrix-Matrix Multiply,
   PageRank, ``apps/mm``, The pagerank algorithm, WIP
   N-Queens, WIP, The n-queens problem, WIP
   GPU API Examples, ``example/gpuapi``, ,

.. note:: This section assumes the Chapel-GPU components are already installed in ``$CHPL_GPU_HOME``. If you have not done so please see :ref:`Building Chapel-GPU`.

Compiling Applications
########################

The example applications in ``chapel-gpu/example`` and ``chapel-gpu/apps`` directory can be build by just doing ``make X``, where ``X`` is either ``cuda``, ``hip``, or ``opencl``. Please be sure to ``source`` the setting script before doing so.

1. Set environment variables

  .. code-block:: bash

     source $CHPL_GPU_HOME/bin/env.sh

2. Compile

   - Example 1: ``chapel-gpu/example``

     .. code-block:: bash

        cd path/to/chapel-gpu/example
        make cuda
        or
        make hip
        or
        make opencl

   - Example 2: ``chapel-gpu/example/gpuapi``

     .. code-block:: bash

        cd path/to/chapel-gpu/example/gpuapi/2d
        make cuda
        or
        make hip

   - Example 3: ``chapel-gpu/apps/stream``

     .. code-block:: bash

        cd path/to/chapel-gpu/apps/stream
        make cuda
        or
        make hip
        or
        make opencl

   .. note:: A baseline implementation for CPUs can be built by doing ``make baseline``.

3. Check the generated executables

   For example, ``make cuda`` in ``apps/vector_copy`` generates the following files:

   .. csv-table::
      :header: "Name", "Description", "Individual make command"
      :widths: 10, 20, 20

      ``vc.baseline``, A baseline implementation for CPUs., ``make baseline``
      ``vc.cuda.gpu``, A GPU-only implmentation w/o the GPUIterator., ``make cuda.gpu``
      ``vc.cuda.hybrid``, The GPUIterator implemenation (single-locale)., ``make cuda.hybrid``
      ``vc.cuda.hybrid.dist``, The GPUIterator implemenation (multi-locale)., ``make cuda.hybrid.dist``
      ``vc.cuda.hybrid.dist.midlow``, The MID-LOW implemenation (multi-locale)., ``make cuda.hybrid.dist.midlow``
      ``vc.cuda.hybrid.dist.mid``, The MID implementation (multi-locale)., ``make cuda.hybrid.dist.mid``


   .. tip:: If you want to compile a specific variant, please do ``make X.Y``, where ``X`` is either ``cuda``, ``hip``, or ``opencl``, and ``Y`` is either ``gpu``, ``hybrid``, ``hybrid.dist``, ``hybrid.dist.midlow``, or ``hybrid.dist.mid``. Please also see the third column above. Also, the MID-LOW and MID variants with OpenCL are currently not supported.

  .. note:: The ``Makefile`` internally uses ``cmake`` to generate a static library from a GPU source program (``vc.cu`` in this case)

Running Applications
#####################

Once you have compiled a Chapel-GPU program, you can run it from the command-line:

.. code-block:: bash

   ./vc.cuda.hybrid

Also, many of the example applications accepts the ``--n`` option, which changes input size, the ``--CPUratio`` (or ``--CPUPercent``) option, which controls the percentage of an iteration space will be executed on CPUs, and the ``--output`` option, which outputs the result arrays. For example:

.. code-block:: bash

   ./vc.cuda.hybrid --n=256 --CPUratio=50 --output=1

For multi-locale execution, please refer to `this document <https://chapel-lang.org/docs/usingchapel/QUICKSTART.html#using-chapel-in-multi-locale-mode>`_.

