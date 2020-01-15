/*
 * Copyright (c) 2019, Rice University
 * Copyright (c) 2019, Georgia Institute of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

module GPUIterator {
    use Time;
    use BlockDist;

    config param debugGPUIterator = false;

    // Utility functions
    inline proc computeSubranges(whole: range(?),
                                 CPUPercent: int(64)) {

      const CPUnumElements = (whole.size * CPUPercent * 1.0 / 100.0) : int(64);
      const CPUhi = (whole.low + CPUnumElements - 1);
      const CPUrange = whole.low..CPUhi;
      const GPUlo = CPUhi + 1;
      const GPUrange = GPUlo..whole.high;

      return (CPUrange, GPUrange);
    }

    inline proc computeChunk(r: range, myChunk, numChunks)
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

    iter createTaskAndYield(param tag: iterKind,
                            r: range(?),
                            CPUrange: range(?),
                            GPUrange: range(?),
                            GPUWrapper: func(int, int, int, void))
      where tag == iterKind.leader {

      if (CPUrange.size == 0) {
        const myIters = GPUrange;
        if (debugGPUIterator) then
          writeln("GPU portion: ", myIters);
        GPUWrapper(myIters.translate(-r.low).first, myIters.translate(-r.low).last, GPUrange.length);
      } else if (GPUrange.size == 0) {
        const numTasks = here.maxTaskPar;
        if (debugGPUIterator) then
          writeln("CPU portion: ", CPUrange, " by ", numTasks, " tasks");
        coforall tid in 0..#numTasks {
          const myIters = computeChunk(CPUrange, tid, numTasks);
          yield (myIters.translate(-r.low),);
        }
      } else {
        cobegin {
          // CPU portion
          {
            const numTasks = here.maxTaskPar;
            if (debugGPUIterator) then
              writeln("CPU portion: ", CPUrange, " by ", numTasks, " tasks");
            coforall tid in 0..#numTasks {
              const myIters = computeChunk(CPUrange, tid, numTasks);
              yield (myIters.translate(-r.low),);
            }
          }
          // GPU portion
          {
            const myIters = GPUrange;
            if (debugGPUIterator) then
              writeln("GPU portion: ", myIters);
            GPUWrapper(myIters.translate(-r.low).first, myIters.translate(-r.low).last, GPUrange.length);
          }
        }
      }
    }

    iter createTaskAndYield(param tag: iterKind,
                            r: range(?),
                            CPUrange: range(?),
                            GPUrange: range(?),
                            GPUWrapper: func(int, int, int, void))
      where tag == iterKind.standalone {

      if (CPUrange.size == 0) {
        const myIters = GPUrange;
        if (debugGPUIterator) then
          writeln("GPU portion: ", myIters);
        GPUWrapper(myIters.translate(-r.low).first, myIters.translate(-r.low).last, GPUrange.length);
      } else if (GPUrange.size == 0) {
        const numTasks = here.maxTaskPar;
        if (debugGPUIterator) then
          writeln("CPU portion: ", CPUrange, " by ", numTasks, " tasks");
        coforall tid in 0..#numTasks {
          const myIters = computeChunk(CPUrange, tid, numTasks);
          for i in myIters do
            yield i;
        }
      } else {
        cobegin {
          // CPU portion
          {
            const numTasks = here.maxTaskPar;
            if (debugGPUIterator) then
              writeln("CPU portion: ", CPUrange, " by ", numTasks, " tasks");
            coforall tid in 0..#numTasks {
              const myIters = computeChunk(CPUrange, tid, numTasks);
              for i in myIters do
                yield i;
            }
          }
          // GPU portion
          {
            const myIters = GPUrange;
            if (debugGPUIterator) then
              writeln("GPU portion: ", myIters);
            GPUWrapper(myIters.translate(-r.low).first, myIters.translate(-r.low).last, GPUrange.length);
          }
        }
      }
    }

    iter createTaskAndYield(r: range(?),
                            CPUrange: range(?),
                            GPUrange: range(?),
                            GPUWrapper: func(int, int, int, void)) {
      halt("This is dummy");
    }

    // leader (block distributed domains)
    iter GPU(param tag: iterKind,
             D: domain,
             GPUWrapper: func(int, int, int, void),
             CPUPercent: int = 0
             )
       where tag == iterKind.leader
       && isRectangularDom(D)
       && D.dist.type <= Block {

      if (debugGPUIterator) {
        writeln("GPUIterator (leader, block distributed)");
      }

      coforall loc in D.targetLocales() do on loc {
        for subdom in D.localSubdomains() {
          const r = subdom.dim(1);
          const portions = computeSubranges(r, CPUPercent);
          for i in createTaskAndYield(tag, 0..0, portions(1), portions(2), GPUWrapper) {
            yield i;
          }
        }
      }
    }

    // follower (block distributed domains)
    iter GPU(param tag: iterKind,
             D: domain,
             GPUWrapper: func(int, int, int, void),
             CPUPercent: int = 0,
             followThis
             )
      where tag == iterKind.follower
      && followThis.size == 1
      && isRectangularDom(D)
      && D.dist.type <= Block {

      const lowBasedIters = followThis(1).translate(D.low);

      if (debugGPUIterator) {
        writeln("GPUIterator (follower, block distributed)");
        writeln("Follower received ", followThis, " as work chunk; shifting to ",
                lowBasedIters);
      }

      for i in lowBasedIters do
        yield i;
    }

    // standalone (block distributed domains)
    iter GPU(param tag: iterKind,
             D: domain,
             GPUWrapper: func(int, int, int, void),
             CPUPercent: int = 0
             )
      where tag == iterKind.standalone
      && isRectangularDom(D)
      && D.dist.type <= Block {

      if (debugGPUIterator) {
        writeln("GPUIterator (standalone distributed)");
      }

      // for each locale
      coforall loc in D.targetLocales() do on loc {
        for subdom in D.localSubdomains() {
          if (debugGPUIterator) then writeln(here, " (", here.name,  ") is responsible for ", subdom);
          const r = subdom.dim(1);
          const portions = computeSubranges(r, CPUPercent);

          for i in createTaskAndYield(tag, 0..0, portions(1), portions(2), GPUWrapper) {
            yield i;
          }
        }
      }
    }

    // serial iterator (block distributed domains)
    iter GPU(D: domain,
             GPUWrapper: func(int, int, int, void),
             CPUPercent: int = 0
             )
      where isRectangularDom(D)
      && D.dist.type <= Block {

      if (debugGPUIterator) {
        writeln("GPUIterator (serial distributed)");
      }
      for i in D {
        yield i;
      }
    }

    // leader (range)
    iter GPU(param tag: iterKind,
             r: range(?),
             GPUWrapper: func(int, int, int, void),
             CPUPercent: int = 0
             )
      where tag == iterKind.leader {

      if (debugGPUIterator) then
	    writeln("In GPUIterator (leader range)");

      const portions = computeSubranges(r, CPUPercent);
      for i in createTaskAndYield(tag, r, portions(1), portions(2), GPUWrapper) {
        yield i;
      }
    }

    // follower
    iter GPU(param tag: iterKind,
             r:range(?),
             GPUWrapper: func(int, int, int, void),
             CPUPercent: int = 0,
             followThis
             )
      where tag == iterKind.follower
      && followThis.size == 1 {

      const lowBasedIters = followThis(1).translate(r.low);

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
             GPUWrapper: func(int, int, int, void),
             CPUPercent: int = 0
             )
  	  where tag == iterKind.standalone {

      if (debugGPUIterator) then
	    writeln("In GPUIterator (standalone)");

      const portions = computeSubranges(r, CPUPercent);
      for i in createTaskAndYield(tag, r, portions(1), portions(2), GPUWrapper) {
        yield i;
      }
    }

    // serial iterators (range)
    iter GPU(r:range(?),
             GPUWrapper: func(int, int, int, void),
             CPUPercent: int = 0
             ) {
      if (debugGPUIterator) then
        writeln("In GPUIterator (serial)");

      for i in r do
        yield i;
    }
}