use GPUIterator;

config const n = 1024;
config const CPUPercent = 50;
config param useGPU = false; // for Mason

var A: [1..n] real(32);
var B: [1..n] real(32);

extern proc vcGPU(A: [] real(32), B: [] real(32), lo: int, hi: int, N: int) where useGPU == true;

// This callback function is called after the GPUIterator
// has computed a subspace for the GPU part
proc GPUCallBack(lo: int, hi: int, N: int) {
  // Note: lo, and hi are 0-origin
  //       so that they can be easily handled by the C side
  if (hi-lo+1 != N) {
    exit();
  }

  if (useGPU == false) {
      // This for loop should be replaced
      // with a function call to the GPU part
      // Since lo and hi are converted to 0-origin,
      // 1 is added to lo and hi in this example
      for i in lo..hi {
          A(i+1) = B(i+1);
      }
  } else {
      vcGPU(A, B, lo, hi, N);
  }
}

B = 1;

// Vector Copy with GPUIterator
forall i in GPU(1..n, GPUCallBack, CPUPercent) {
  A(i) = B(i);
}

// verify
if (A.equals(B)) {
  writeln("Verified");
} else {
  writeln("Not Verified");
  exit();
}