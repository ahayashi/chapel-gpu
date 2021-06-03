use GPUAPI;
use SysCTypes;
use CPtr;

extern proc kernel(dA: c_void_ptr, nRows: size_t, nCols: size_t, dpitch: size_t);

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
init();

var dA: c_void_ptr;
var hpitch: size_t = D.dim(1).size:size_t * c_sizeof(A.eltType);
var dpitch: size_t;

MallocPitch(dA, dpitch, hpitch, D.dim(0).size: size_t);
writeln("MID-LOW: pitch on the host:", hpitch);
writeln("MID-LOW: pitch on the device: ", dpitch);

Memcpy2D(dA, dpitch, c_ptrTo(A), hpitch, hpitch, D.dim(0).size: size_t, 0);
kernel(dA, D.dim(0).size: size_t, D.dim(1).size: size_t, dpitch);
DeviceSynchronize();
Memcpy2D(c_ptrTo(A), hpitch, dA, dpitch, hpitch, D.dim(0).size: size_t, 1);

// Verify
if (A.equals(V)) {
    writeln("MID-LOW Verified");
} else {
    writeln("MID-LOW Not Verified");
}

// MID (not pitched)
init();
var dA2 = new GPUArray(A);
writeln("MID (pitch=false) pitch on the host:", dA2.hpitch);
writeln("MID (pitch=false) pitch on the device: ", dA2.dpitch);
if (dA2.hpitch != dA2.dpitch) {
    writeln("Error: the pitch on the host must be the same as that on the device when pitch=false");
    exit();
}

dA2.toDevice();
kernel(dA2.dPtr(), D.dim(0).size: size_t, D.dim(1).size: size_t, dA2.dpitch);
DeviceSynchronize();
dA2.fromDevice();

// Verify
if (A.equals(V)) {
    writeln("MID (pitch=false) Verified");
} else {
    writeln("MID (pitch=false) Not Verified");
}

// MID (pitched)
init();
var dA3 = new GPUArray(A, true);
writeln("MID (pitch=true) pitch on the host:", dA3.hpitch);
writeln("MID (pitch=true) pitch on the device: ", dA3.dpitch);
dA3.toDevice();
kernel(dA3.dPtr(), D.dim(0).size: size_t, D.dim(1).size: size_t, dA3.dpitch);
DeviceSynchronize();
dA3.fromDevice();

// Verify
if (A.equals(V)) {
    writeln("MID (pitch=true) Verified");
} else {
    writeln("MID (pitch=true) Not Verified");
}
