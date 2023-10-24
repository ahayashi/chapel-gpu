use GPUAPI;
use CTypes;

extern proc kernel(dA: c_ptr(void), n: int);

var D = {0..31};
var A: [D] int;
var V: [D] int; // for verification

// initialization proc
proc initialize() {
    for i in D {
        A[i] = i;
    }
    V = A + 1;
}

// MID-LOW
// dA is a linearized 1D GPU array
initialize();

var dA: c_ptr(void);
var size: c_size_t = A.size:c_size_t * c_sizeof(A.eltType);
Malloc(dA, size);
Memcpy(dA, c_ptrTo(A), size, 0);
kernel(dA, D.dim(0).size);
DeviceSynchronize();
Memcpy(c_ptrTo(A), dA, size, 1);

// Verify
if (A.equals(V)) {
    writeln("MID-LOW Verified");
} else {
    writeln("MID-LOW Not Verified");
}

// MID
initialize();

var dA2 = new GPUArray(A);
dA2.toDevice();

kernel(dA2.dPtr(), D.dim(0).size);
DeviceSynchronize();
dA2.fromDevice();

// Verify
if (A.equals(V)) {
    writeln("MID Verified");
} else {
    writeln("MID Not Verified");
}
