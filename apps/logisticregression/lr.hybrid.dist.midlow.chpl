use Time;
use Math;
use ReplicatedDist;
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
config const nFeatures = 32: int;
config const nSamples = 32: int;
config const nIters = 32: int;
config const CPUratio = 0: int;
config const numTrials = 1: int;
config const output = 0: int;
config param verbose = false;
config const reduction = false;

////////////////////////////////////////////////////////////////////////////////
/// Global Arrays
////////////////////////////////////////////////////////////////////////////////
// For now, these arrays are global so the arrays can be seen from CUDAWrapper
// TODO: Explore the possiblity of declaring the arrays and CUDAWrapper
//       in the main proc (e.g., by using lambdas)
const Space1 = {1..nSamples, 1..nFeatures};
const ReplicatedSpace1 = Space1 dmapped replicatedDist();
var X: [ReplicatedSpace1] real(32);

const Space2 = {1..nSamples};
const ReplicatedSpace2 = Space2 dmapped replicatedDist();
var Y: [ReplicatedSpace2] real(32);

const Space3 = {1..nFeatures};
const ReplicatedSpace3 = Space3 dmapped replicatedDist();
var Wcurr: [ReplicatedSpace3] real(32);

var D: domain(1) dmapped blockDist(boundingBox = {1..nFeatures}) = {1..nFeatures};
var W: [D] real(32);
var alpha = 0.1 : real(32);

////////////////////////////////////////////////////////////////////////////////
/// C Interoperability
////////////////////////////////////////////////////////////////////////////////
extern proc LaunchLR(X: c_ptr(void), Y: c_ptr(void), W: c_ptr(void), Wcurr: c_ptr(void), alpha: real(32), nSamples: int, nFeatures: int, lo: int, hi: int, N: int);

// CUDAWrapper is called from GPUIterator
// to invoke a specific CUDA program (using C interoperability)
proc CUDAWrapper(lo: int, hi: int, N: int) {
  if (verbose) {
	writeln("In CUDAWrapper2(), launching the CUDA kernel with a range of ", lo, "..", hi, " (Size: ", N, ")");
  }
  ref lW = W.localSlice(lo .. hi);
  if (verbose) { ProfilerStart(); }
  var dX, dY, dWcurr, dW: c_ptr(void);
  Malloc(dX, X.size:c_size_t * c_sizeof(X.eltType));
  Malloc(dY, Y.size:c_size_t * c_sizeof(Y.eltType));
  Malloc(dWcurr, Wcurr.size:c_size_t * c_sizeof(Wcurr.eltType));
  Malloc(dW, lW.size:c_size_t * c_sizeof(lW.eltType));
  Memcpy(dX, c_ptrTo(X.replicand(here)), X.size:c_size_t * c_sizeof(X.eltType), 0);
  Memcpy(dY, c_ptrTo(Y.replicand(here)), Y.size:c_size_t * c_sizeof(Y.eltType), 0);
  Memcpy(dWcurr, c_ptrTo(Wcurr.replicand(here)), Wcurr.size:c_size_t * c_sizeof(Wcurr.eltType), 0);
  LaunchLR(dX, dY, dW, dWcurr, alpha, nSamples, nFeatures, lo, hi, N);
  DeviceSynchronize();
  Memcpy(c_ptrTo(lW), dW, lW.size:c_size_t * c_sizeof(lW.eltType), 1);
  Free(dX);
  Free(dY);
  Free(dW);
  Free(dWcurr);
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
  writeln("Logistic Regression: CPU/GPU Execution (using GPUIterator)");
  writeln("nSamples :", nSamples, " nFeatures :",  nFeatures);
  writeln("CPU Percent: ", CPUratio);
  writeln("nGPUs: ", nGPUs);  
  writeln("nTrials: ", numTrials);
  writeln("output: ", output);
  writeln("reduction: ", reduction);

  printLocaleInfo();

  var execTimes: [1..numTrials] real;
  var execTimes2: [1..numTrials] real;  
  for trial in 1..numTrials {
    if (false) {
      forall i in D {
        W(i) = 0: real(32);
      }
      coforall loc in Locales do on loc {
          for i in 1..nSamples {
            Y(i) = (i % 2): real(32);
            for j in 1..nFeatures {
              if (j != 0) {
                X(i, j) = (i % 2): real(32);
              } else {
                X(i, j) = 1;
              }
            }
          }
        }
    } else {
      forall i in D {
        W(i) = 0: real(32);
      }
      coforall loc in Locales do on loc {
          forall i in 1..nSamples {
            Y(i) = i: real(32);
            for j in 1..nFeatures {
              if (j != 0) {
                X(i, j) = j: real(32);
              } else {
                X(i, j) = j : real(32);
              }
            }
          }
        }
    }

    const startTime = timeSinceEpoch().totalSeconds();
	for ite in 1..nIters {
      coforall loc in Locales {
        on loc {
          Wcurr = W;
        }
      }
      const start = timeSinceEpoch().totalSeconds();
      forall i in GPU(D, CUDAWrapper, CPUratio) {
		var err = 0: real(32);
		for s in 1..nSamples {
          var arg = 0: real(32);
          if (reduction) {
            arg = (+ reduce (Wcurr(1..nFeatures) * X(s, 1..nFeatures)));
          } else {
            for f in 1..nFeatures {
              arg += Wcurr(f) * X(s, f);
            }
          }
          var hypo = 1 / (1 + exp(-arg));
          err += (hypo - Y(s)) * X(s, i);
		}
		W(i) = Wcurr(i) - alpha * err;
      }
      execTimes2(trial) = timeSinceEpoch().totalSeconds() - start;
	}
	execTimes(trial) = timeSinceEpoch().totalSeconds() - startTime;
	if (output) {
      writeln(W);
	}
  }
  printResults(execTimes);
  printResults(execTimes2);  
}
