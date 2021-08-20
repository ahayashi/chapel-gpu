use DynGPUIterator;
use Time;

var D: domain(1) = {1..64};

proc GPUWrapper(lo: int, hi: int, N: int) {
    writeln("GPU =  ", {lo..hi});
}

forall i in DynGPU(D, GPUWrapper, 4) {
    writeln("CPU = ", i);
    sleep(1);
}