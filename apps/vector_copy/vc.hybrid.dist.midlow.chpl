use Time;

////////////////////////////////////////////////////////////////////////////////
/// GPUIterator
////////////////////////////////////////////////////////////////////////////////
use GPUIterator;
use GPUAPI;
use BlockDist;
use CTypes;

////////////////////////////////////////////////////////////////////////////////
/// Runtime Options
////////////////////////////////////////////////////////////////////////////////
config const n = 32: int;
config const CPUratio = 0: int;
config const numTrials = 1: int;
config const output = 0: int;
config param verbose = false;

////////////////////////////////////////////////////////////////////////////////
/// Global Arrays
////////////////////////////////////////////////////////////////////////////////
// For now, these arrays are global so the arrays can be seen from CUDAWrapper
// TODO: Explore the possiblity of declaring the arrays and CUDAWrapper
//       in the main proc (e.g., by using lambdas)
var D: domain(1) dmapped blockDist(boundingBox = {1..n}) = {1..n};
var A: [D] real(32);
var B: [D] real(32);

////////////////////////////////////////////////////////////////////////////////
/// C Interoperability
////////////////////////////////////////////////////////////////////////////////
extern proc LaunchVC(A: c_ptr(void), B: c_ptr(void), N: c_size_t);

// CUDAWrapper is called from GPUIterator
// to invoke a specific CUDA program (using C interoperability)
proc CUDAWrapper(lo: int, hi: int, N: int) {
  if (verbose) {
    var device, count: int(32);
    GetDevice(device);
    GetDeviceCount(count);
	writeln("In CUDAWrapper(), launching the CUDA kernel with a range of ", lo, "..", hi, " (Size: ", N, "), GPU", device, " of ", count, " @", here);
  }
  ref lA = A.localSlice(lo .. hi);
  ref lB = B.localSlice(lo .. hi);
  if (verbose) { ProfilerStart(); }
  var dA, dB: c_ptr(void);
  var size: c_size_t = (lA.size:c_size_t * c_sizeof(lA.eltType));
  Malloc(dA, size);
  Malloc(dB, size);
  Memcpy(dB, c_ptrTo(lB), size, 0);
  LaunchVC(dA, dB, N: c_size_t);
  DeviceSynchronize();
  Memcpy(c_ptrTo(lA), dA, size, 1);
  Free(dA);
  Free(dB);
  if (verbose) { ProfilerStop(); }
}

////////////////////////////////////////////////////////////////////////////////
/// Utility Functions
////////////////////////////////////////////////////////////////////////////////
proc printResults(execTimes) {
  const totalTime = + reduce execTimes,
	avgTime = totalTime / numTrials,
	minTime = min reduce execTimes;
  writeln("Execution time:");
  writeln("  tot = ", totalTime);
  writeln("  avg = ", avgTime);
  writeln("  min = ", minTime);
}

proc printLocaleInfo() {
  for loc in Locales {
    writeln(loc, " info: ");
    const numGPUs = loc.gpus.size;
    if (numGPUs != 0) {
      for gpuID in 0..#numGPUs {
        const gpu = loc.gpus[gpuID];
        writeln("\t Subloc: ", gpuID);
        writeln("\t Name: ", gpu);
        writeln("\t maxTaskPar: ", gpu.maxTaskPar);
      }
    } else {
      writeln("\t Name: ", loc);
      writeln("\t maxTaskPar: ", loc.maxTaskPar);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Chapel main
////////////////////////////////////////////////////////////////////////////////
proc main() {
  writeln("Vector Copy: CPU/GPU Execution (using GPUIterator)");
  writeln("Size: ", n);
  writeln("CPU ratio: ", CPUratio);
  writeln("nGPUs: ", nGPUs);
  writeln("nTrials: ", numTrials);
  writeln("output: ", output);

  printLocaleInfo();

  var execTimes: [1..numTrials] real;
  for trial in 1..numTrials {
	forall i in D {
      A(i) = 0: real(32);
      B(i) = i: real(32);
	}

	const startTime = timeSinceEpoch().totalSeconds();
	forall i in GPU(D, CUDAWrapper, CPUratio) {
      A(i) = B(i);
	}
	execTimes(trial) = timeSinceEpoch().totalSeconds() - startTime;
	if (output) {
      writeln(A);
      for i in 1..n {
        if (A(i) != B(i)) {
          writeln("Verification Error");
          exit();
        }
      }
      writeln("Verified");
	}
  }
  printResults(execTimes);
}
