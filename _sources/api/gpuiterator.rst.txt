.. default-domain:: chpl

===============
GPUIterator
===============

.. iterfunction:: iter GPU(c: range(?), GPUCallBack: func(int, int, int, void), CPUPercent: int = 0)

   :arg c: The range to iterate over. The length of the range must be greater
           than zero.
   :type c: `range(?)`           

   :arg GPUCallBack: The reference to a Chapel function that is invoked after the iterator has computed a subrange for the GPU portion. It must take three integers: ``lo:int, hi:int, n:int``, where ``lo`` and ``hi`` are the lower and the upper bound of the GPU portion respectively, and ``n`` is ``hi-lo+1``.
   :type GPUCallBack: `func(int, int, int, void)`

   :arg CPUPercent: The percentage of the iteration space will be executed on the CPU. The default number for it is zero, meaning the whole itreration space goes to the GPU side.
   :type CPUPercent: `int`

   :yields: Indices in the CPU portion of the range ``c``. 


.. iterfunction:: iter GPU(D: domain, GPUCallBack: func(int, int, int, void), CPUPercent: int = 0)

   :arg D: The domain to iterate over. The length of the range must be greater
           than zero. It must be a rectangular domain. Also, if ``D`` is ``dmapped``, it must be ``BlockDist``.
   :type D: `domain`           

   :arg GPUCallBack: The reference to a Chapel function that is invoked after the iterator has computed a subrange for the GPU portion. It must take three integers: ``lo:int, hi:int, n:int``, where ``lo`` and ``hi`` are the lower and the upper bound of the GPU portion respectively, and ``n`` is ``hi-lo+1``.
   :type GPUCallBack: `func(int, int, int, void)`

   :arg CPUPercent: The percentage of the iteration space will be executed on the CPU. The default number for it is zero, meaning the whole itreration space goes to the GPU side.
   :type CPUPercent: `int`

   :yields: Indices in the CPU portion of the range ``D``. 
            
