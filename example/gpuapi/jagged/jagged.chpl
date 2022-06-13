use GPUAPI;
use CTypes;

extern proc kernelLOW(a: [] c_ptr(int), b: [] c_size_t, n: int);
extern proc kernelMIDLOW(a: c_ptr(c_void_ptr), n: int);

class C {
  var n: int;
  proc init(_n: int) { n = _n;  }
  var x: [0..#n] int;
}

var Cs = [new C(256), new C(512)];
var Vs = [new C(256), new C(512)];
const N = Cs.size;

proc init() {
  for e in Cs {
    for i in e.x.domain {
      e.x[i] = i;
    }
  }
  for e in Vs {
    for i in e.x.domain {
      e.x[i] = i+1;
    }
  }
}

proc verify(prefix) {
  var verified = true;
  for i in 0..#N {
    if (Cs[i].x.equals(Vs[i].x) == false) {
      verified = false;
    }
  }
  if (verified) {
    writeln(prefix + " Verified");
  } else {
    writeln(prefix + " Not Verified");
  }
}

// LOW
writeln("[LOW]");
{
    init();
    ref ptrs = [c_ptrTo(Cs[0].x), c_ptrTo(Cs[1].x)];
    ref sizes = [Cs[0].x.size:c_size_t*c_sizeof(int), Cs[1].x.size:c_size_t*c_sizeof(int)];
    kernelLOW(ptrs, sizes, N);
    verify("LOW");
}

// MIDLOW
writeln("[MIDLOW]");
{
  init();
  var dA: [0..#N] c_void_ptr;
  var dAs: c_ptr(c_void_ptr);
  for i in 0..#N {
    const size = Cs[i].x.size:c_size_t*c_sizeof(int);
    Malloc(dA[i], size);
    Memcpy(dA[i], c_ptrTo(Cs[i].x), size, 0);
  }
  const size = N: c_size_t * c_sizeof(c_ptr(c_void_ptr));
  Malloc(dAs, size);
  Memcpy(dAs, c_ptrTo(dA), size, 0);

  kernelMIDLOW(dAs, N);
  DeviceSynchronize();
  for i in 0..#N {
    const size = Cs[i].x.size:c_size_t*c_sizeof(int);
    Memcpy(c_ptrTo(Cs[i].x), dA[i], size, 1);
  }
  // Verification
  verify("LOWMID");
}

// MID
writeln("[MID: multiple args]");
{
    init();
    var dAs = new GPUJaggedArray(Cs[0].x, Cs[1].x);
    dAs.toDevice();
    kernelMIDLOW(dAs.dPtr(), N);
    dAs.fromDevice();
    // Verification
    verify("MID: multiple args");
}

writeln("[MID: promoted]");
{
    init();
    var dAs = new GPUJaggedArray(Cs.x);
    dAs.toDevice();
    kernelMIDLOW(dAs.dPtr(), N);
    dAs.fromDevice();
    // Verification
    verify("MID: promoted");
}

