use GPUIterator;

config const n = 1024;
config const CPUPercent = 50;

var A: [1..n] real(32);
var B: [1..n] real(32);

for i in 1..n {
  A(i) = 0: real(32);
  B(i) = i: real(32);
}

// This callback function is called after the GPUIterator
// has computed a subspace for the GPU part
var GPUCallBack = lambda(lo: int, hi: int, nElems: int) {
  // Note: lo, and hi are 0-origin
  //       so that they can be easily handled by the C side
  if (hi-lo+1 != nElems) {                
    exit();
  }
  // This for loop should be replaced
  // with a function call to the GPU part
  // Since lo and hi are converted to 0-origin,
  // 1 is added to lo and hi in this example
  for i in lo..hi {
    A(i+1) = B(i+1);
  }
};

// Vector Copy with GPUIterator
forall i in GPU(1..n, GPUCallBack, CPUPercent) {
  A(i) = B(i);
}

// verify
for i in 1..n {
  if (A(i) != i) {
    halt("Verification Error");
  }
}

writeln("Verified");