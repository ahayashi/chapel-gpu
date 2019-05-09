module GPUIterator {
    use Time;
    use BlockDist;
    use DSIUtil;

    config param debugGPUIterator = false;
    config param distOpt = false;

    // leader (block distributed domains)
    iter GPU(param tag: iterKind,
             D: domain,
             GPUWrapper: func(int, int, int, void),
             CPUratio: int = 0
             )
      where tag == iterKind.leader
      && isRectangularDom(D)
      && D.dist.type <= Block {

      if (debugGPUIterator) then writeln("GPUIterator (leader)");

      var dist = D.dist;
      var whole = D.whole;
      var locDoms = D.locDoms;
      type idxType = D.dist.idxType;
      param rank = D.dist.rank;

      const maxTasks = dist.dataParTasksPerLocale;
      const ignoreRunning = dist.dataParIgnoreRunningTasks;
      const minSize = dist.dataParMinGranularity;
      const wholeLow = whole.low;
      const hereId = here.id;
      const hereIgnoreRunning = if here.runningTasks() == 1 then true
        else ignoreRunning;

      // for each locale
      coforall locDom in locDoms do on locDom {
          const myIgnoreRunning = if here.id == hereId then hereIgnoreRunning else ignoreRunning;

          // Use the internal function for untranslate to avoid having to do
          // extra work to negate the offset
          type strType = chpl__signedType(idxType);
          const tmpBlock = locDom.myBlock.chpl__unTranslate(wholeLow);
          var locOffset: rank*idxType;
          for param i in 1..tmpBlock.rank {
            const stride = tmpBlock.dim(i).stride;
            if stride < 0 && strType != idxType then
              halt("negative stride not supported with unsigned idxType");
            // (since locOffset is unsigned in that case)
            locOffset(i) = tmpBlock.dim(i).first / stride:idxType;
          }
          if (debugGPUIterator) then writeln(locDom.locale, " (", locDom.locale.name,  ") is responsible for ", tmpBlock);

          const r = tmpBlock;

          if (CPUratio >= 0) {
              const CPUnumElements = (r.size * (CPUratio*1.0/100.0)): int;

              const CPUhi = (r.low + CPUnumElements - 1);
              const CPUrange = r.low..CPUhi;
              const GPUlo = CPUhi + 1;
              const GPUrange = GPUlo..r.high;

              if (debugGPUIterator) then
                writeln(here);

              coforall locs in 0..1 {
                if (locs == 0) {
                  // CPU parallel iterator
                  var c = here.getChild(locs);
                  const numTasks = c.maxTaskPar;
                  if (debugGPUIterator) then
                    writeln("\tSubloc ", locs, " owns ", CPUrange, " and decompose it into ", numTasks, " tasks");
                  coforall tid in 0..#numTasks {
                    const myIters = computeChunk(CPUrange, tid, numTasks);
                    if (debugGPUIterator) then
                      writeln("\tCPU's task ", tid, " owns ", myIters);
                    yield (myIters,);
                  }
                } else {
                  // GPU parallel iterator
                  const myIters = GPUrange;
                  if (debugGPUIterator) then
                    writeln("\tSubloc ", locs, " owns ", myIters);
                  /* Current Version: importing hand-coded version */
                  GPUWrapper(myIters.translate(whole.low).first, myIters.translate(whole.low).last, GPUrange.length);
                }
              }
          }
        }
    }

    // leader (range)
    iter GPU(param tag: iterKind,
             r: range(?),
             CUDAWrapper: func(int, int, int, void),
             CPUratio: int = 0
             )
      where tag == iterKind.leader {

      const numSublocs = here.getChildCount();

      if (debugGPUIterator) then
	    writeln("In GPUIterator, creating ", numSublocs, " parallel iterators (CPU/GPU)");

      if (CPUratio >= 0) {
        const CPUnumElements = (r.length * (CPUratio*1.0/100.0)): int;

        const CPUhi = (r.low + CPUnumElements - 1);
        const CPUrange = r.low..CPUhi;
        const GPUlo = CPUhi + 1;
        const GPUrange = GPUlo..r.high;

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
            CUDAWrapper(myIters.translate(-r.low).first, myIters.translate(-r.low).last, GPUrange.length);
          }
        }
      }
    }

    // follower
    iter GPU(param tag: iterKind,
             D,
             GPUWrapper: func(int, int, int, void),
             CPUratio: int,
             followThis
             )
      where tag == iterKind.follower
      && followThis.size == 1 {

      const lowBasedIters = followThis(1).translate(D.low);

      if (debugGPUIterator) {
        writeln("GPUIterator (follower)");
        writeln("Follower received ", followThis, " as work chunk; shifting to ",
                lowBasedIters);
      }

      for i in lowBasedIters do
        yield i;
    }

    // standalone (range)
    iter GPU(param tag: iterKind,
             r: range(?),
             CUDAWrapper: func(int, int, int, void),
             CPUratio: int = 0
             )
  	  where tag == iterKind.standalone {

      const numSublocs = here.getChildCount();

      if (debugGPUIterator) then
	    writeln("In GPUIterator, creating ", numSublocs, " parallel iterators (CPU/GPU)");

      const CPUnumElements = (r.length * (CPUratio*1.0/100.0)): int;

      const CPUhi = (r.low + CPUnumElements - 1);
      const CPUrange = r.low..CPUhi;
      const GPUlo = CPUhi + 1;
      const GPUrange = GPUlo..r.high;

      if (CPUratio == 0) {
        const myIters = GPUrange;
        if (debugGPUIterator) then
          writeln("Subloc 1 owns ", myIters);
        /* Current Version: importing hand-coded version */
        CUDAWrapper(myIters.translate(-r.low).first, myIters.translate(-r.low).last, GPUrange.length);
      } else if (CPUratio == 100) {
        // CPU parallel iterator
        var c = here.getChild(0);
        const numTasks = c.maxTaskPar;
        if (debugGPUIterator) then
          writeln("Subloc 0 owns ", CPUrange, " and decompose it into ", numTasks, " tasks");
        coforall tid in 0..#numTasks {
          const myIters = computeChunk(CPUrange, tid, numTasks);
          if (debugGPUIterator) then
            writeln("CPU's task ", tid, " owns ", myIters);
          for i in myIters do
            yield i;
        }
      } else if (CPUratio > 0 && CPUratio < 100) {
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
            CUDAWrapper(myIters.translate(-r.low).first, myIters.translate(-r.low).last, GPUrange.length);
          }
        }
      } else {
        var execTimes: [0..100 by 10] real;
        for ratio in 0..100 by 10 {
          const startTime = getCurrentTime();
          const CPUnumElements = (r.length * (ratio*1.0/100.0)): int;

          const CPUhi = (r.low + CPUnumElements - 1);
          const CPUrange = r.low..CPUhi;
          const GPUlo = CPUhi + 1;
          const GPUrange = GPUlo..r.high;

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
              CUDAWrapper(myIters.translate(-r.low).first, myIters.translate(-r.low).last, GPUrange.length);
            }
          }
          execTimes(ratio) = getCurrentTime() - startTime;
        }
        for ratio in 0..100 by 10 {
          writeln("CPUratio = ", ratio, ", ", execTimes(ratio));
        }
      }
    }

    // standalone (block distributed domains)
    iter GPU(param tag: iterKind,
             D: domain,
             GPUWrapper: func(int, int, int, void),
             CPUratio: int = 0
             )
      where tag == iterKind.standalone
      && isRectangularDom(D)
      && D.dist.type <= Block {

      if (debugGPUIterator) then writeln("GPUIterator (standalone distributed)");

      var dist = D.dist;
      var whole = D.whole;
      var locDoms = D.locDoms;
      type idxType = D.dist.idxType;
      param rank = D.dist.rank;

      const maxTasks = dist.dataParTasksPerLocale;
      const ignoreRunning = dist.dataParIgnoreRunningTasks;
      const minSize = dist.dataParMinGranularity;
      const wholeLow = whole.low;
      const hereId = here.id;
      const hereIgnoreRunning = if here.runningTasks() == 1 then true
        else ignoreRunning;

      // for each locale
      coforall locDom in locDoms do on locDom {
          const myIgnoreRunning = if here.id == hereId then hereIgnoreRunning else ignoreRunning;

          // Use the internal function for untranslate to avoid having to do
          // extra work to negate the offset
          type strType = chpl__signedType(idxType);
          const tmpBlock = locDom.myBlock.chpl__unTranslate(wholeLow);
          var locOffset: rank*idxType;
          for param i in 1..tmpBlock.rank {
            const stride = tmpBlock.dim(i).stride;
            if stride < 0 && strType != idxType then
              halt("negative stride not supported with unsigned idxType");
            // (since locOffset is unsigned in that case)
            locOffset(i) = tmpBlock.dim(i).first / stride:idxType;
          }
          if (debugGPUIterator) then writeln(locDom.locale, " (", locDom.locale.name,  ") is responsible for ", tmpBlock);

          const r = tmpBlock;
          const CPUnumElements = (r.size * (CPUratio*1.0/100.0)): int;

          const CPUhi = (r.low + CPUnumElements - 1);
          const CPUrange = r.low..CPUhi;
          const GPUlo = CPUhi + 1;
          const GPUrange = GPUlo..r.high;

          if (CPUratio == 0) {
            const myIters = GPUrange;
            if (debugGPUIterator) then
              writeln("\tSubloc  1 owns ", myIters);
            /* Current Version: importing hand-coded version */
            GPUWrapper(myIters.translate(whole.low).first, myIters.translate(whole.low).last, GPUrange.length);
          } else if (CPUratio == 100) {
            // CPU parallel iterator
            var c = here.getChild(0);
            const numTasks = c.maxTaskPar;
            if (debugGPUIterator) then
              writeln("Subloc 0 owns ", CPUrange, " and decompose it into ", numTasks, " tasks");
            coforall tid in 0..#numTasks {
              const myIters = computeChunk(CPUrange, tid, numTasks).translate(wholeLow);
              if (debugGPUIterator) then
                writeln("\tCPU's task ", tid, " owns ", myIters);
              for i in myIters {
                yield i;
              }
            }
          } else if (CPUratio > 0 && CPUratio < 100) {
              coforall locs in 0..1 {
                if (locs == 0) {
                  // CPU parallel iterator
                  var c = here.getChild(locs);
                  const numTasks = c.maxTaskPar;
                  if (debugGPUIterator) then
                    writeln("\tSubloc ", locs, " owns ", CPUrange, " and decompose it into ", numTasks, " tasks");
                  coforall tid in 0..#numTasks {
                    const myIters = computeChunk(CPUrange, tid, numTasks);
                    if (debugGPUIterator) then
                      writeln("\tCPU's task ", tid, " owns ", myIters);
                    for i in myIters.translate(whole.low) {
                      yield i;
                    }
                  }
                } else {
                  // GPU parallel iterator
                  const myIters = GPUrange;
                  if (debugGPUIterator) then
                    writeln("\tSubloc ", locs, " owns ", myIters);
                  /* Current Version: importing hand-coded version */
                  GPUWrapper(myIters.translate(whole.low).first, myIters.translate(whole.low).last, GPUrange.length);
                }
              }
          }
        }
    }

    // serial iterator
    iter GPU(D,
             GPUWrapper: func(int, int, int, void),
             CPUratio: int = 0
             ) {
      if (debugGPUIterator) then writeln("GPUIterator (serial)");
      for i in D {
        yield i;
      }
    }

    iter GPU(r: range(?),
             CUDAWrapper: func(int, int, int, void),
             CPUratio: int=0) {
      for i in r.low..r.high do
	    yield i;
    }

    proc computeChunk(r: range, myChunk, numChunks)
      where r.stridable == false {

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