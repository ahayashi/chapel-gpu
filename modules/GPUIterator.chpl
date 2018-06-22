module GPUIterator {
    param debugGPUIterator = true;

    iter GPU(param tag: iterKind,
	      lo: int, hi: int,
	      CUDAWrapper: func(int, int, int, void),
	      CPUratio: int = 0)
	where tag == iterKind.standalone {
	const numSublocs = here.getChildCount();
	
	if (debugGPUIterator) then
	    writeln("In GPUIterator, creating ", numSublocs, " parallel iterators (CPU/GPU)");
	
	const range = lo..hi;
	
	const CPUnumElements = (range.length * (CPUratio*1.0/100.0)): int;
	
	const CPUhi = (lo + CPUnumElements - 1);
	const CPUrange = lo..CPUhi;    
	const GPUlo = CPUhi + 1;
	const GPUrange = GPUlo..hi;
	
	coforall locs in 0..1 {
	    if (locs == 0) {
		// CPU parallel iterator
		var c = here.getChild(locs);
		const numTasks = c.maxTaskPar;
		if (debugGPUIterator) then
		    writeln("Subloc ", locs, " owns ", CPUrange, " and decompose it into ", numTasks, " tasks");
		coforall tid in 0..#numTasks {
		    const myIters = computeChunk(CPUrange, tid, numTasks);
		    if (debugGPUIterator) then
			writeln("CPU's task ", tid, " owns ", myIters);
		    for i in myIters do
			yield i;
		}
		
	    } else {
		// GPU parallel iterator
		const myIters = GPUrange;
		if (debugGPUIterator) then
		    writeln("Subloc ", locs, " owns ", myIters);
		/* Current Version: importing hand-coded version */
		CUDAWrapper(myIters.translate(-lo).first, myIters.translate(-lo).last, GPUrange.length);
		/* Future Version: compiler's GPU code generation */
//	        for i in myIters do
//		  yield i;
	    }
	}
    }
    
    iter GPU(lo: int, hi: int,
	     CUDAWrapper: func(int, int, int, void),
	     CPUratio: int=100) {
	for i in lo..hi do
	    yield i;
    }
    
    proc computeChunk(r: range, myChunk, numChunks) where r.stridable == false {
	const numElems = r.length;
	const elemsPerChunk = numElems/numChunks;
	const mylow = r.low + elemsPerChunk*myChunk;
	if (myChunk != numChunks - 1) {
	    return mylow..#elemsPerChunk;
	} else {
	    return mylow..r.high;
	}
    }    
}
