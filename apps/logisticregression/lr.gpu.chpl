use Time;

////////////////////////////////////////////////////////////////////////////////
/// Runtime Options
////////////////////////////////////////////////////////////////////////////////
config const nFeatures = 32: int;
config const nSamples = 32: int;
config const nIters = 32: int;
config const numTrials = 1: int;
config const output = 0: int;
config param verbose = false;

////////////////////////////////////////////////////////////////////////////////
/// Global Arrays
////////////////////////////////////////////////////////////////////////////////
// For now, these arrays are global so the arrays can be seen from CUDAWrapper
// TODO: Explore the possiblity of declaring the arrays and CUDAWrapper
//       in the main proc (e.g., by using lambdas)
var X: [1..nSamples, 1..nFeatures] real(32);
var Y: [1..nSamples] real(32);
var W: [1..nFeatures] real(32);
var Wcurr: [1..nFeatures] real(32);
var alpha = 0.1 : real(32);

////////////////////////////////////////////////////////////////////////////////
/// C Interoperability
////////////////////////////////////////////////////////////////////////////////
extern proc lrCUDA1(W: [] real(32), Wcurr: [] real(32), lo: int, hi: int, N: int);
extern proc lrCUDA2(X: [] real(32), Y: [] real(32), W: [] real(32), Wcurr: [] real(32), alpha: real(32), nSamples: int, nFeatures: int, lo: int, hi: int, N: int);

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
  writeln("Logistic Regression: GPU Only");
  writeln("nSamples :", nSamples, " nFeatures :",  nFeatures);
  writeln("nTrials: ", numTrials);
  writeln("output: ", output);

  printLocaleInfo();

  var execTimes: [1..numTrials] real;
  for trial in 1..numTrials {
    if (false) {
      for i in 1..nFeatures {
        W(i) = 0: real(32);
      }
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
    } else {
      forall i in 1..nFeatures {
        W(i) = 0: real(32);
      }
      for i in 1..nSamples {
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

	const startTime = timeSinceEpoch().totalSeconds();
	for ite in 1..nIters {
      lrCUDA1(W, Wcurr, 0, nFeatures-1, nFeatures);
      lrCUDA2(X, Y, W, Wcurr, alpha, nSamples, nFeatures, 1, nFeatures, nFeatures);
	}
	execTimes(trial) = timeSinceEpoch().totalSeconds() - startTime;
	if (output) {
      writeln(W);
	}
  }
  printResults(execTimes);
}
