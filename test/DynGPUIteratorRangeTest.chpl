use DynGPUIterator;
use Time;

proc GPUWrapper(lo: int, hi: int, N: int) {
    writeln("GPU =  ", {lo..hi});
}

forall i in DynGPU(1..64, GPUWrapper, 4) {
    writeln("CPU = ", i);
    sleep(1);
}