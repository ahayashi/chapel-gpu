

module DynGPUIterator {
    use GPUIterator;
    use ChapelLocks, DSIUtil;
    use Time;

    // serial iterator
    iter DynGPU(D: domain,
             GPUWrapper,
             chunkSize: int = 4
        )
        where  isRectangularDom(D) {
        for i in D do yield i;
    }

    // serial iterators (range)
    iter DynGPU(r:range(?),
                GPUWrapper,
                chunkSize: int = 0
        ) {
        for i in r do
            yield i;
    }

    // dynamic parallel standalone iterator
    iter DynGPU(param tag: iterKind,
             D: domain,
             GPUWrapper,
             chunkSize: int = 4
        )
        where tag == iterKind.standalone
        && isRectangularDom(D)
    {
        const numChunks: int;
        if D.idxType == uint(64) then
            numChunks = divceil(D.size, chunkSize:uint(64)): int;
        else
            numChunks = divceilpos(D.size:int(64), chunkSize): int;

        type rType=D.type;

        // We're going to have to densify at some point, might as well
        // do it early and make range slicing easier.
        const remain:rType=densify(D.dim(0),D.dim(0));

        const numCPUTasks = here.maxTaskPar;
        const numGPUTasks = nGPUs;
        const nTasks = numCPUTasks + numGPUTasks;

        var moreWork : atomic bool = true;
        var curChunkIdx : atomic int = 0;

        coforall tid in 0..#nTasks with (const in remain) {
            while moreWork.read() {
                const chunkIdx = curChunkIdx.fetchAdd(1);
                const low = chunkIdx * chunkSize; /* remain.low is 0, stride is 1 */
                const high: low.type;
                if chunkSize >= max(low.type) - low then
                    high = max(low.type);
                else
                    high = low + chunkSize-1;

                if chunkIdx >= numChunks {
                    break;
                } else if high >= remain.high {
                    moreWork.write(false);
                }
                const current:rType = remain(low .. high);
                if (debugGPUIterator) then writeln("Parallel dynamic Iterator. Working at tid ", tid, " with range ", unDensify(current,D), " yielded as ", current);
                if (tid < numCPUTasks) {
                    yield (current,);
                } else {
                    GPUWrapper(low, high, high-low+1);
                }
            }
        }
    }

    // dynamic parallel standalone iterator
    iter DynGPU(param tag: iterKind,
             r: range(?),
             GPUWrapper,
             chunkSize: int = 4
        )
        where tag == iterKind.standalone {
        var D: domain(1) = r;
        forall i in DynGPU(D, GPUWrapper, chunkSize) {
            yield i;
        }
    }
}