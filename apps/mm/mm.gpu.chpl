use Time;

////////////////////////////////////////////////////////////////////////////////
/// Runtime Options
////////////////////////////////////////////////////////////////////////////////
config const n = 32: int;
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
var A: [1..n, 1..n] real(32);
var B: [1..n, 1..n] real(32);
var C: [1..n, 1..n] real(32);

////////////////////////////////////////////////////////////////////////////////
/// C Interoperability
////////////////////////////////////////////////////////////////////////////////
extern proc mmCUDA(A: [] real(32), B: [] real(32), C: [] real(32), N: int, lo: int, hi: int, GPUN: int, tiled: int);

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
  writeln("Matrix Multiplication: GPU Only");
  writeln("Size: ", n, "x", n);
  writeln("nTrials: ", numTrials);
  writeln("tiled: ", tiled);
  writeln("output: ", output);

  printLocaleInfo();

  var execTimes: [1..numTrials] real;
  for trial in 1..numTrials {
	for i in 1..n {
      for j in 1..n {
		A(i, j) = i: real(32);
		B(i, j) = i: real(32);
		C(i, j) = 0: real(32);
      }
	}

	const startTime = getCurrentTime();
	mmCUDA(A, B, C, n*n, 0, n*n-1, n*n, tiled);
	execTimes(trial) = getCurrentTime() - startTime;
	if (output) {
      writeln(C);
	}
  }
  printResults(execTimes);
}
