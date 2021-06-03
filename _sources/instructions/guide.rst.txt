=============================================
Guide to Write GPU programs
=============================================

General Guidelines
###################

In general, GPU programs should include typical host and device operations including device memory (de)allocations, data transfers, and kernel invocations. Depending on the abstraction level you choose, some of these operations can be written in a Chapel-user-friendly way:

.. list-table::
   :widths: 15 15 15 15
   :header-rows: 1

   * - Level
     - MID-level
     - MID-LOW-level
     - LOW-level
   * - Kernel Invocation
     - CUDA/HIP
     - CUDA/HIP
     - CUDA/HIP/OpenCL
   * - Memory (de)allocations
     - Chapel (MID)
     - Chapel (MID-LOW)
     - CUDA/HIP/OpenCL
   * - Data transfers
     - Chapel (MID)
     - Chapel (MID-LOW)
     - CUDA/HIP/OpenCL


.. seealso::

   * :ref:`Writing MID-level programs`
   * :ref:`MID-level API Reference`
   * :ref:`Writing MID-LOW-level programs`
   * :ref:`MID-LOW-level API Reference`
   * :ref:`Writing LOW-level (GPUIterator Only) programs`

.. note:: LOW/MID-LOW/MID levels can interoperate with each other.


Writing GPU program
#######################################


The design and implementation of a CUDA/HIP/OpenCL program that is supposed to be called from the callback function is completely up to you. However, please be aware that it can be called multiple times (i.e., the number of GPUs per locale * the number of locales) as the GPUIterator automatically and implicitly handles multiple- GPUs and locales. We'd highly recommend writing your GPU program in a way that is 1) device neutral (no device setting call) and 2) flexibile to change in iteration spaces -i.e., ``start`` and ``end``  (including data allocations and transfers).

.. Data Transfers
.. ***************

.. .. code-block:: chapel

..   forall i in GPU(1..n, GPUCallBack) {
..      A(i) = B(i);
..   }


.. Write a GPU program that is flexible to adapt to different iteration spaces.

.. is GPU ID neutral, where [DEFINITION], which improve the portability of your GPU program significantly.
