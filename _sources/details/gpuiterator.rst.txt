.. default-domain:: chpl

===========
GPUIterator
===========

Overview
############
A primary goal of this module is to provide an appropriate interface between Chapel and accelerator programs such that expert accelerator programmers can explore different variants in a portable way (i.e., CPU-only, GPU-only, X% for CPU + Y% for GPU on a single or multiple CPU+GPU node(s)). To address these challenges, here we introduce a Chapel module, ``GPUIterator``, which facilitates invoking a user-written GPU program from Chapel. Since `Chapel's data-parallel loops <https://chapel-lang.org/docs/users-guide/datapar/forall.html>`_ (``forall``) fit well with GPU execution, the ``GPUIterator`` is designed to be invoked in a ``forall`` loop. Consider the following ``STREAM`` code:

.. code-block:: chapel
                
  forall i in 1..n {
      A(i) = B(i) + alpha * C(i);
  }



Assuming a GPU version of ``STREAM`` is ready (``streamCUDA`` below), the user can wrap the original iteration space in ``GPU()`` with two additional arguments: ``GPUCallBack`` is a callback function that is invoked after the module has computed a subrange for the GPU portion by using ``CPUPercent``:

.. code-block:: chapel
   :linenos:
      
   // A GPUIterator version
   extern proc streamCUDA(A: [] real(32), B:[] real(32), C:[] real(32),
                          alpha: real(32), lo: int, hi: int, N: int); 
   var GPUCallBack = lambda(lo: int, hi: int, N: int) {
     // call the GPU program with a range of lo..hi
     streamCUDA(A, B, C, alpha, lo, hi, N);
   };
   CPUPercent = 50; // CPU 50% + GPU 50% in this case
   forall i in GPU(1..n, GPUCallBack, CPUPercent) {
     A(i) = B(i) + alpha * C(i);
   }

  
It is worth noting that ``GPUIterator`` supports multi-GPUs execution and `multi-locale execution <https://chapel-lang.org/docs/users-guide/locality/compilingAndExecutingMultiLocalePrograms.html>`_. For multi-GPUs execution, the module automatically detects the numbers of GPUs per node (or accept a user-specified number), and invokes the callback function for each GPU, which can be done without any modification to the code above. For multi-locale execution, the iterator accepts a `block distributed domain <https://chapel-lang.org/docs/primers/distributions.html#block-and-distribution-basics>`_, which allows the user to run the code above on multiple CPUs+GPUs nodes with minimal modifications. 

Why GPUIterator?
##################
Chapel offers `the C interoperability feature <https://chapel-lang.org/docs/master/technotes/extern.html>`_, which allows the user to invoke C/C++ functions from their Chapel programs. In the context of GPU programming in Chapel, the user typically prepares a GPU version of a ``forall`` loop written in CUDA/HIP/OpenCL and invokes it using the interoperability feature. For example, consider the following baseline ``forall`` implementation that performs ``STREAM``:

.. code-block:: chapel
   :linenos:                
                    
   // Chapel file
   var A: [1..n] real(32);
   var B: [1..n] real(32);
   var C: [1..n] real(32);
   var alpha: real(32) = 3.0;
   forall i in 1..n {
     A(i) = B(i) + alpha * C(i);
   }

Assuming ``streamCUDA()``, which is a full CUDA/HIP/OpenCL implementation of the ``forall``, is available, here is what the GPU version looks like:

.. code-block:: chapel
   :linenos:
      
   // Chapel file
   // Declare an external C/C++ function which performs STREAM on GPUs
   extern proc streamCUDA(A: [] real(32), B:[] real(32), C:[] real(32),
                          alpha: real(32), lo: int, hi: int, N: int);

   var A: [1..n] real(32);
   var B: [1..n] real(32);
   var C: [1..n] real(32);
   var alpha: real(32);
   streamCUDA(A, B, C, alpha, 1, n, n);

   
.. code-block:: c
   :linenos:
      
   // Separate C file
   void streamCUDA(float *A, float *B, float *C,
                   float alpha, int start, int end, int size) {
   // A full GPU implementation of STREAM (CUDA/HIP/OpenCL)
   // 1. device memory allocations
   // 2. host-to-device data transfers
   // 3. GPU kernel compilations (if needed)
   // 4. GPU kernel invocations
   // 5. device-to-host data transfers
   // 6. clean up
   // Note: A[0] and B[0] here corresponds to
   // A(1) and B(1) in the Chapel part respectively
   }


The key difference is that the original ``forall`` loop is replaced with the function call to the native function that includes typical host and device operations including device memory (de)allocations, data transfers, and kernel invocations.

Unfortunately, the source code is not very portable particularly when the user wants to explore different configurations to get higher performance. One scenario is that, since GPUs are not always faster than CPUs (and vice versa), the user has to be juggling ``forall`` with ``streamCUDA()`` depending on the data size and the complexity of computations (e.g., by commenting in/out each version).

One intuitive workaround would be to put an ``if`` statement to decide whether to use which version (CPUs or GPUs):

.. code-block:: chapel
   :linenos:                

   if (cond) {
     forall i in 1..n { // STREAM }
   } else {
     streamCUDA(...);
   }

However, this raises another problem: it is still not very portable when the user wants to do 1) multi-locale CPU+GPU execution, and 2) advanced workload distributions such as hybrid execution of the CPU and GPU versions. Specifically, WITHOUT the module, the user has to write the following code:

.. code-block:: chapel
   :linenos:

   // WITHOUT the GPUIterator module (no hybrid execution)
   // suppose D is a block distributed domain
   if (cond) {
     forall i in D { ... }
   } else {
     coforall loc in Locales {
       on loc {
         coforall GPUID in 0..#nGPUs {
           var lo = ...; // needs to be computed manually
           var hi = ...; // needs to be computed manually
           var localA = A.localSlice(lo..hi);
           ...
           // GPUID needs to be manually set before streamCUDA() is called
           streamCUDA(localA, ...); 
         }
       }
     }
   }


WITH the module, again, the code is much simpler and more portable:

.. code-block:: chapel
   :linenos:               

   // WITH the GPUIterator module
   // suppose D is a block distributed domain
   var GPUCallBack = lambda(lo: int, hi: int, N: int) {
     // call the GPU program with a range of lo..hi
     // lo..hi is automatically computed
     // the module internally and automatically sets GPUID
     streamCUDA(A.localSlice(lo..hi), ...);
   };
   CPUPercent = 50; // CPU 50% + GPU 50% in this case
   forall i in GPU(D, GPUCallBack, CPUPercent) {
     A(i) = B(i) + alpha * C(i);
   }



Further Readings
################

* GPUIterator: bridging the gap between Chapel and GPU platforms. Akihiro Hayashi, Sri Raj Paul, Vivek Sarkar, The ACM SIGPLAN 6th Annual Chapel Implementers and Users Workshop (CHIUW), June 2019. (co-located with PLDI2019/ACM FCRC2019) `DOI <https://dl.acm.org/doi/10.1145/3329722.3330142>`_.

