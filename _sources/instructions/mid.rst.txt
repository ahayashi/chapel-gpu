.. default-domain:: chpl

=============================================
Writing MID-level programs
=============================================

MID-level API
######################

To reiterate, the biggest motivation for introducing ``LOW-MID`` and ``MID`` -level GPU API is moving some of low-level GPU operations to the Chapel-level. Consider the following GPU callback function and C function:

.. code-block:: chapel
   :caption: vc.hybrid.chpl

   // lo, hi, and N are automatically computed by the GPUIterator
   proc GPUCallBack(lo: int, hi: int, N: int) {
     vcCUDA(A, B, lo, hi, N);
   }

.. code-block:: c
   :caption: vc.cu

   extern "C" {
     void vcCUDA(float* A, float *B, int start, int end, int GPUN) {
       float *dA, *dB;
       cudaMalloc(&dA, sizeof(float) * GPUN);
       cudaMalloc(&dB, sizeof(float) * GPUN);
       cudaMemcpy(dB, B + start, sizeof(float) * GPUN, cudaMemcpyHostToDevice);
       vc<<<ceil(((float)GPUN)/1024), 1024>>>(dA, dB, GPUN);
       cudaDeviceSynchronize();
       cudaMemcpy(A + start, dA, sizeof(float) * GPUN, cudaMemcpyDeviceToHost);
       cudaFree(dA);
       cudaFree(dB);
     }
   }

At the MID-level, most of the CUDA/HIP/OpenCL-level 1) device memory allocation, 2) device synchronization, and 3) data transfer can be written in Chapel. Also, unlike the LOW-MID level, the MID-level API is more Chapel programmer-friendly, where you can allocate GPU memory using the ``new`` keyword and no longer need to directly manipulate C types. Here is an example program written with the MID-level API:


.. code-block:: chapel
   :caption: vc.hybrid.chpl

   proc GPUCallBack(lo: int, hi: int, N: int) {
     // n * sizeof(int) will be automatically allocated onto the device
     var dA = new GPUArray(A);
     var dB = new GPUArray(B);
     dB.toDevice();
     LaunchVC(dA.dPtr(), dB.dPtr(), N: size_t);
     dA.fromDevice();
     free(dA, dB);
   }

.. tip:: The MID-level API can interoperate with the LOW-MID-level API.

.. seealso:: :ref:`MID-level API Reference`

