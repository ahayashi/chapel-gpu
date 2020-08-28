.. default-domain:: chpl
                    
=============================================
Writing LOW-MID-level programs
=============================================

LOW-MID-level API
######################

The biggest motivation for introducing ``LOW-MID`` and ``MID`` -level GPU API is moving some of low-level GPU operations to the Chapel-level. Consider the following GPU callback function and C function:

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

At the LOW-MID-level, most of the CUDA/HIP/OpenCL-level 1) device memory allocation, 2) device synchronization, and 3) data transfer can be written in Chapel. However, it's worth noting that this level of abstraction only provides thin wrapper functions for the CUDA/HIP/OpenCL-level API functions, which requires you to directly manipulate C types like ``c_void_ptr`` and so on. The LOW-MID is helpful particularly when you want to fine-tune the use of GPU API, but still want to stick with Chapel. Here is an example program written with the LOW-MID-level API:

.. code-block:: chapel
   :caption: vc.hybrid.chpl

   proc GPUCallBack(lo: int, hi: int, N: int) {
     var dA, dB: c_void_ptr;
     var size: size_t = (lA.size:size_t * c_sizeof(lA.eltType));
     Malloc(dA, size);
     Malloc(dB, size);
     Memcpy(dB, c_ptrTo(lB), size, 0);
     LaunchVC(dA, dB, N: size_t);
     DeviceSynchronize();
     Memcpy(c_ptrTo(lA), dA, size, 1);
     Free(dA);
     Free(dB);     
   }

.. tip:: The LOW-MID-level API can interoperate with the MID-level API.

.. seealso:: :ref:`LOW-MID-level API Reference`
