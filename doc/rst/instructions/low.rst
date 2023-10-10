.. default-domain:: chpl

=============================================
Writing LOW-level (GPUIterator Only) programs
=============================================

Here we provide a step-by-step guide for utilizing the ``GPUIterator`` module using a simple Chapel program (``vector copy``) in single-locale and multiple-locale scenarios.

Single-locale version
######################

In this single-locale scenario, you are supposed to create and edit one Chapel source file and one CUDA source file: ``vc.hybrid.chpl`` and ``vc.cu``.

1. Import the GPUIterator module

   First, import the module using the ``use`` keyword:

   .. code-block:: chapel
      :caption: vc.hybrid.chpl

      use GPUIterator;

2. Declare Chapel arrays

   Then, create two Chapel arrays, ``A`` and ``B``, which will be used for the copy operation:

   .. code-block:: chapel
      :caption: vc.hybrid.chpl
      :emphasize-lines: 3,4,5

      use GPUIterator;

      config const n = 32: int;
      var A: [1..n] real(32);
      var B: [1..n] real(32);



   .. tip:: It's wise to define ``n`` as `a configurable constant <https://chapel-lang.org/docs/users-guide/base/configs.html>`_, which can be overridden on the command line (e.g., ``./vc --n=1024``).

3. Import your GPU program

   a. Write a GPU program

      It is worth noting that the design and implementation of the GPU program is completely your choice. Please also see :ref:`Guide to Write GPU programs`. Here is one working vector copy example with CUDA:

   .. code-block:: c
      :caption: vc.cu

      __global__ void vc(float *dA, float *dB, int N) {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < N) {
	      dA[id] = dB[id];
        }
      }

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



   .. note:: For the presentation purposes, any error checking is omitted. A complete program can be found in ``apps`` directory.



   b. Declare it as an external function

      Use `Chapel's C interoperability feature <https://chapel-lang.org/docs/technotes/extern.html>`_ to declare ``vcCUDA()`` as an external function.

   .. code-block:: chapel
      :caption: vc.hybrid.chpl
      :emphasize-lines: 7

      use GPUIterator;

      config const n = 32: int;
      var A: [1..n] real(32);
      var B: [1..n] real(32);

      extern proc vcCUDA(A: [] real(32), B: [] real(32), lo: int, hi: int, N: int);


   .. note:: More details on the C interoperability feature can be found `here <https://chapel-lang.org/docs/technotes/extern.html>`_.


4. Write a GPU callback function

   The GPU callback function is supposed to be invoked from the GPUIterator with an automatically computed subrange (``lo`` and ``hi``). In this example, we call the external function ``vcCUDA`` with the two global arrays (``A`` and ``B``), the subrange (``lo`` and ``hi``), plus the number of elements (``N = size(lo..hi)``).

   .. code-block:: chapel
      :caption: vc.hybrid.chpl
      :emphasize-lines: 9,10,11,12

      use GPUIterator;

      config const n = 32: int;
      var A: [1..n] real(32);
      var B: [1..n] real(32);

      extern proc vcCUDA(A: [] real(32), B: [] real(32), lo: int, hi: int, N: int);

      // lo, hi, and N are automatically computed by the GPUIterator
      proc GPUCallBack(lo: int, hi: int, N: int) {
        vcCUDA(A, B, lo, hi, N);
      }

.. _callback:

   It is worth noting that there will be multiple calls to ``GPUCallBack()`` when the number of GPUs is greater than one. Internally, the GPUIterator detects the number of GPUs within a locale, then automatically computes a subrange for each GPU, and creates a separate task that is responsible for each GPU. This design keeps the callback function simple and independent from GPU ID. The table below illustrates how ``GPUCallBack()`` is called when ``n=1024, nLocales=1, nGPUs=2``:

   .. list-table:: n=1024, nLocales=1, nGPUs=2
      :widths: 15 15 15 15
      :header-rows: 1

      * -
        - Locales[0]
        -
        -
      * -
        - CPUs
        - GPU0
        - GPU1
      * - ``lo..hi``
        - ``1..512``
        - ``512..767``
        - ``768..1024``
      * - ``GPUCallBack(lo,hi,N);``
        - N/A
        - ``GPUCallBack(512,767,256);``
        - ``GPUCallBack(768,1024,256);``

   .. tip:: The number of GPUs can be overridden by giving the `--nGPUs=n` option (two dashes) on the command line

   .. note::

      1. Writing GPU ID dependent code in a callback function can be also done using the ``GetDevice`` function of the GPUAPI :

      .. code-block:: chapel

         use GPUAPI;
         proc GPUCallBack(lo: int, hi:int, N:int) {
           var id;
           GetDevice(id);
           if (id == 0) { ... }
           else if ...
         }

         
      2. While the use of a lambda function would be more productive and elegant, we'd recommend writing a Chapel function for the callback since the lambda support in Chapel is still early.

      .. code-block:: chapel

         var GPUCallBack = lambda(lo: int, hi:int, N:int) { vcCUDA(A, B, lo, hi, N); };
         forall i in GPU(1..n, GPUCallback) { ... }


      If the this lambda version does not work, try `this workaround <https://github.com/chapel-lang/chapel/issues/8351>`_:
         
      .. code-block:: chapel


         record Lambda {
           proc this(lo:int, hi:int, N:int) { vcCUDA(A, B, lo, hi, N); }
         }
         var GPUCallBack = new Lambda();
         forall i in GPU(1..n, GPUCallback) { ... }


5. Invoke the ``GPU()`` iterator in a ``forall`` loop

   When writing a ``forall`` loop, simply wrap the iteration space (``1..n``) in ``GPU()`` and give the callback function (``GPUCallBack``). Here is a complete program with output verification:

   .. code-block:: chapel
      :caption: vc.hybrid.chpl
      :emphasize-lines: 15-18

      use GPUIterator;

      config const n = 32: int;
      var A: [1..n] real(32);
      var B: [1..n] real(32);

      extern proc vcCUDA(A: [] real(32), B: [] real(32), lo: int, hi: int, N: int);

      proc GPUCallBack(lo: int, hi: int, N: int) {
        vcCUDA(A, B, lo, hi, N);
      }

      B = 1;

      forall i in GPU(1..n, GPUCallBack) {
        // CPU Version
        A(i) = B(i);
      }

      if (A.equals(B)) {
        writeln("Verified");
      } else {
        writeln("Not Verified");
      }


6. Compile and Run

   See :doc:`Compiling and running <compile>`

Multi-locale version
######################

In the multi-locale scenario, you are supposed to update ``vc.hybrid.chpl`` slightly, but you can keep the GPU program (``vc.cu``) unchanged.

0. Copy ``vc.hybrid.chpl`` to ``vc.hybrid.dist.chpl``

1. Add ``BlockDist`` module and replace the range with a block-distributed domain

   .. code-block:: chapel
      :caption: vc.hybrid.dist.chpl
      :emphasize-lines: 2

      use GPUIterator
      use BlockDist;


   Then, declare two Chapel arrays with a block-distributed domain ``D``.

   .. code-block:: chapel
      :caption: vc.hybrid.dist.chpl
      :emphasize-lines: 2-4

      config const n = 32: int;
      var D: domain(1) dmapped blockDist(boundingBox = {1..n}) = {1..n};
      var A: [D] real(32);
      var B: [D] real(32);
      // var A: [1..n] real(32); /* single locale version */
      // var B: [1..n] real(32); /* single locale version */


2. Update ``GPUCallBack``

   .. code-block:: chapel
      :caption: vc.hybrid.dist.chpl
      :emphasize-lines: 3-7

      // lo, hi, and N are automatically computed by the GPUIterator
      proc GPUCallBack(lo: int, hi: int, N: int) {
        // the first element of lA is lA(lo), which corresponds to A[0] in the vcCUDA part.
        ref lA = A.localSlice(lo..hi);
        // the first element of lB is lB(lo), which corresponds to B[0] in the vcCUDA part.        
        ref lB = B.localSlice(lo..hi); 
        vcCUDA(lA, lB, 0, hi-lo, N);
        //vcCUDA(A, B, lo, hi, N); /* single locale version */
      }


   While the code looks pretty much similar to the single-locale version, since the two arrays are distributed, the following two additional things need to be done:

   a. Using ``localSlice()`` API

      .. code-block:: chapel

         // for GPU X on locale Y, (locale- and device-neutral)
         ref lA = A.localSlice(lo..hi);
                   

      Similar to the single-locale + multiple GPUs case discussed `above <callback_>`_,  multiple instances of ``GPUCallBack()`` will be invoked for each GPU on different locales. However, you can still write the callback in a way that is locale and GPU ID independent by utilizing Chapel's ``localSlice(d: domain)`` API (`link <https://chapel-lang.org/docs/builtins/ChapelArray.html#ChapelArray.localSlice>`_). Essentially, feeding the automatically computed subrange (``lo..hi``) to the API returns a proper slice of a distributed array in a specific instance of ``GPUCallBack()``.
      

   b. Updating the arguments to ``vcCUDA()``

      .. code-block:: chapel

         // call to the external GPU program
         vcCUDA(lA, lB, 0, hi-lo, N);

         
      Let us first explain how the local reference (say ``lA``) can be accessed in the GPU program (``vcCUDA``). To give you a concrete example, suppose ``n=2048, nLocales=2, CPUPercent=50``, in which ``A(1..1024)`` resides on `Locale 0`, and ``A(1025..2048)`` resides on `Locale 1`. The table below summarizes how ``lA`` corresponds to the C array (``A``) in each instance of the callback:
      
      .. list-table:: n=2048, nLocales=2, nGPUs=2
         :widths: 15 15 15 15 15 15 15
         :header-rows: 1

         * -
           - Locales[0]
           -
           -
           - Locales[1]
           -
           -
         * -
           - CPUs
           - GPU0
           - GPU1
           - CPUs
           - GPU0
           - GPU1
         * - ``lo..hi``
           - ``1..512``
           - ``513..768``
           - ``769..1024``
           - ``1025..1536``
           - ``1537..1792``
           - ``1793..2048``
         * - ``GPUCallBack(lo,hi,N);``
           - N/A
           - ``GPUCallBack(513,768,256);``
           - ``GPUCallBack(769,1024,256);``
           - N/A
           - ``GPUCallBack(1537,1792,256);``
           - ``GPUCallBack(1793,2048,256);``
         * - ``lA = A.localSlice(lo..hi)``
           - N/A
           - ``A.localSlice(513..768);``
           - ``A.localSlice(769..1024);``
           - N/A
           - ``A.localSlice(1537..1792);``
           - ``A.localSlice(1793..2048);``             
         * - ``A[0]`` in ``vcCUDA`` corresponds to
           - N/A
           - ``lA(513)``
           - ``lA(769)``
           - N/A
           - ``lA(1537)``
           - ``lA(1793)``


      Notice that ``A[0]`` in ``vcCUDA(float *A, ...)`` corresponds to the first element of the local slice, which is why the third argument is zero (= ``start``) and thr fourth argument is ``hi-lo`` (= ``end``).

3. Update ``GPU()``

   Finally, give the distributed domain (``D``) to ``GPU()``:

   .. code-block:: chapel
      :caption: vc.hybrid.dist.chpl

      forall i in GPU(D, GPUCallBack) {
      //forall i in GPU(1..n, GPUCallBack) {
        // CPU Version
        A(i) = B(i);
      }


4. Compile and Run

   See :doc:`Compiling and running <compile>`
      
