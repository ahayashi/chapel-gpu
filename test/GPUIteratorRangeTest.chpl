use GPUIterator;

const n = 1024;
const CPUPercent = 50;
var A: [1..n] real(32);
var B: [1..n] real(32);

for CPUPercent in (0, 25, 50, 75, 100) {

  for i in 1..n {
    A(i) = -1: real(32);
    B(i) = i: real(32);
  }

  var GPUCallBack = lambda(lo: int, hi: int, nElems: int) {
    if (hi-lo+1 != nElems) {
      exit();
    }
    // this is where an external GPU function is supposed to be invoked
    // for testing purpose, do nothing
  };

  // Vector Copy with GPUIterator
  forall i in GPU(1..n, GPUCallBack, CPUPercent) {
    A(i) = B(i);
  }

  // verify
  for i in 1..n {
    if (i <= n * CPUPercent/100) {
      if (A(i) != i) {
        exit();
      }
    } else {
      if (A(i) != 0) {
        exit();
      }
    }
  }
}
