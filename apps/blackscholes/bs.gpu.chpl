use Time;

////////////////////////////////////////////////////////////////////////////////
/// Runtime Options
////////////////////////////////////////////////////////////////////////////////
config const n = 32: int;
config const numTrials = 1: int;
config const output = 0: int;
config param verbose = false;

////////////////////////////////////////////////////////////////////////////////
/// Global Arrays
////////////////////////////////////////////////////////////////////////////////
// For now, these arrays are global so the arrays can be seen from CUDAWrapper
// TODO: Explore the possiblity of declaring the arrays and CUDAWrapper
//       in the main proc (e.g., by using lambdas)
var rand: [1..n] real(32);
var put: [1..n] real(32);
var call: [1..n] real(32);

////////////////////////////////////////////////////////////////////////////////
/// C Interoperability
////////////////////////////////////////////////////////////////////////////////
extern proc bsCUDA(rand: [] real(32), put: [] real(32), call: [] real(32), lo: int, hi: int, N: int);

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
  writeln("BlackScholes: GPU Only");
  writeln("Size: ", n);
  writeln("nTrials: ", numTrials);
  writeln("output: ", output);

  printLocaleInfo();

  const S_LOWER_LIMIT = 10.0: real(32);
  const S_UPPER_LIMIT = 100.0: real(32);
  const K_LOWER_LIMIT = 10.0: real(32);
  const K_UPPER_LIMIT = 100.0: real(32);
  const T_LOWER_LIMIT = 1.0: real(32);
  const T_UPPER_LIMIT = 10.0: real(32);
  const R_LOWER_LIMIT = 0.01: real(32);
  const R_UPPER_LIMIT = 0.05: real(32);
  const SIGMA_LOWER_LIMIT = 0.01: real(32);
  const SIGMA_UPPER_LIMIT = 0.10: real(32);

  var execTimes: [1..numTrials] real;
  for trial in 1..numTrials {
	for i in 1..n {
      rand(i) = (i: real(32) / n): real(32);
	}

	const startTime = timeSinceEpoch().totalSeconds();
	bsCUDA(rand, put, call, 0, n-1, n);
	execTimes(trial) = timeSinceEpoch().totalSeconds() - startTime;
	if (output) {
      writeln(call);
      writeln(put);
	}
  }
  printResults(execTimes);
}
