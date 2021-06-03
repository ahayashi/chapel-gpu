use GPUAPI;
use SysCTypes;
use CPtr;

extern proc kernel(dA: c_void_ptr, nRows: int, nCols: int);

var D = {0..8, 0..8};
var A: [D] int;
var V: [D] int; // for verification

// initialization proc
proc init() {
    for (i, j) in D {
        A[i, j] = (i+1)*10 + j;
    }
    V = A + 1;
}

// MID-LOW
// dA is a linearized 1D GPU array
init();

var dA: c_void_ptr;
var size: size_t = A.size:size_t * c_sizeof(A.eltType);
Malloc(dA, size);
Memcpy(dA, c_ptrTo(A), size, 0);
kernel(dA, D.dim(0).size, D.dim(1).size);
DeviceSynchronize();
Memcpy(c_ptrTo(A), dA, size, 1);

// Verify
if (A.equals(V)) {
    writeln("MID-LOW Verified");
} else {
    writeln("MID-LOW Not Verified");
}

// MID
init();

var dA2 = new GPUArray(A);
dA2.toDevice();

kernel(dA2.dPtr(), D.dim(0).size, D.dim(1).size);
DeviceSynchronize();
dA2.fromDevice();

// Verify
if (A.equals(V)) {
    writeln("MID Verified");
} else {
    writeln("MID Not Verified");
}