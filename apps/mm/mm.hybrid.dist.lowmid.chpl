use Time;
use ReplicatedDist;
////////////////////////////////////////////////////////////////////////////////
/// GPUIterator
////////////////////////////////////////////////////////////////////////////////
use GPUIterator;
use GPUAPI;
use BlockDist;
use SysCTypes;

////////////////////////////////////////////////////////////////////////////////
/// Runtime Options
////////////////////////////////////////////////////////////////////////////////
config const n = 32: int;
config const CPUratio = 0: int;
config const numTrials = 1: int;
config const tiled = 0;
config const output = 0: int;
config param verbose = false;

////////////////////////////////////////////////////////////////////////////////
/// Global Arrays
////////////////////////////////////////////////////////////////////////////////
// For now, these arrays are global so the arrays can be seen from CUDAWrapper
// TODO: Explore the possiblity of declaring the arrays and CUDAWrapper
//       in the main proc (e.g., by using lambdas)
const S = {1..n, 1..n};
const RS = S dmapped Replicated();
var D: domain(1) dmapped Block(boundingBox = {1..n*n}) = {1..n*n};

var A: [D] real(32);
var B: [RS] real(32);
var C: [D] real(32);

////////////////////////////////////////////////////////////////////////////////
/// C Interoperability
////////////////////////////////////////////////////////////////////////////////
extern proc LaunchMM(A: c_void_ptr, B: c_void_ptr, C: c_void_ptr, N: int, lo:int, hi:int, GPUN: int, tiled: int);

// CUDAWrapper is called from GPUIterator
// to invoke a specific CUDA program (using C interoperability)
proc CUDAWrapper(lo: int, hi: int, N: int) {
  if (verbose) {
	writeln("In CUDAWrapper(), launching the CUDA kernel with a range of ", lo, "..", hi, " (Size: ", N, ")");
  }
  //if(tiled) {
  //  assert(N/n>=32 && (N/n)%32==0, "should use multiples of 32 rows in GPU when tiled");
  //}
  assert(N%n == 0, "should offload full rows to GPU");
  ref lA = A.localSlice(lo .. hi);
  ref lC = C.localSlice(lo .. hi);
  assert(lA.size == lC.size);

  if (verbose) { ProfilerStart(); }
  var dA, dB, dC: c_void_ptr;

  //writeln("lA.size: ", lA.size, " B.size: ", B.size);
  Malloc(dA, lA.size:size_t * c_sizeof(lA.eltType));
  Malloc(dB, B.size:size_t  * c_sizeof(B.eltType));
  Malloc(dC, lC.size:size_t * c_sizeof(lC.eltType));

  Memcpy(dA, c_ptrTo(lA), lA.size:size_t * c_sizeof(lA.eltType), 0);
  Memcpy(dB, c_ptrTo(B),  B.size:size_t  * c_sizeof(B.eltType),  0);

  LaunchMM(dA, dB, dC, n*n, 0, hi-lo, N, tiled);
  DeviceSynchronize();
  Memcpy(c_ptrTo(lC), dC, lC.size:size_t * c_sizeof(lC.eltType), 1);

  Free(dA);
  Free(dB);
  Free(dC);
  if (verbose) { ProfilerStop(); }

  //mmCUDA(lA, B, lC, n*n, 0, hi-lo, N, tiled);
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
    const numSublocs = loc.getChildCount();
    if (numSublocs != 0) {
      for sublocID in 0..#numSublocs {
        const subloc = loc.getChild(sublocID);
        writeln("\t Subloc: ", sublocID);
        writeln("\t Name: ", subloc);
        writeln("\t maxTaskPar: ", subloc.maxTaskPar);
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
  writeln("Matrix Multiplication: CPU/GPU Execution (using GPUIterator)");
  writeln("Size: ", n, "x", n);
  writeln("CPU ratio: ", CPUratio);
  writeln("nGPUs: ", nGPUs);
  writeln("nTrials: ", numTrials);
  writeln("tiled: ", tiled);
  writeln("output: ", output);

  printLocaleInfo();

  var execTimes: [1..numTrials] real;
  for trial in 1..numTrials {
    coforall loc in Locales do on loc {
      forall i in 1..n {
        forall j in 1..n {
          var e: int = (i-1)*n+(j-1)+1;
          A(e) = (i*1.0/1000): real(32);
          B(i, j) = (i*1.0/1000): real(32);
          C(e) = 0: real(32);
        }
      }
    }

	const startTime = getCurrentTime();
	// TODO: Consider using a 2D iterator
	forall e in GPU(D, CUDAWrapper, CPUratio) {
      var i: int = (e - 1) / n + 1;
      var j: int = (e - 1) % n + 1;
      var sum: real(32) = C(e);
      for k in 1..n {
		sum += A((i-1)*n+k) * B(k, j);
      }
      C(e) = sum;
	}
	execTimes(trial) = getCurrentTime() - startTime;
	if (output) {
      writeln(reshape(C, {1..n, 1..n}));
	}
  }
  printResults(execTimes);
}
