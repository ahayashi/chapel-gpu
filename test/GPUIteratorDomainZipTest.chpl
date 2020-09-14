use GPUIterator;

const n = 1024;
var A: [1..n] real(32);
var B: [1..n] real(32);

for CPUPercent in (0, 25, 50, 75, 100) {

  for i in 1..n {
    A(i) = -1: real(32);
    B(i) = i: real(32);
  }

  var GPUCallBack = lambda(lo: int, hi: int, nElems: int) {
    if (hi-lo+1 != nElems) {
      exit(1);
    }
    // this is where an external GPU function is supposed to be invoked
    // for testing purpose, do nothing
  };

  // Vector Copy with GPUIterator
  var D: domain(1) = {1..n};
  forall (_, a, b) in zip(GPU(D, GPUCallBack, CPUPercent), A, B) {
    a = b;
  }

  // verify
  for i in 1..n {
    if (i <= n * CPUPercent/100) {
      if (A(i) != i) {
        exit(1);
      }
    } else {
      if (A(i) != -1) {
        exit(1);
      }
    }
  }
  writeln("CPUPercent: ", CPUPercent, " (Verified)");
}
