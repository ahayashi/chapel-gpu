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
  writeln("Logistic Regression: Baseline");
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

	const startTime = getCurrentTime();
	for ite in 1..nIters {
      forall i in 1..nFeatures {
		Wcurr(i) = W(i);
      }
      forall i in 1..nFeatures {
		var err = 0: real(32);
		for s in 1..nSamples {
          var arg = 0: real(32);
          for f in 1..nFeatures {
			arg += Wcurr(f) * X(s, f);
          }
          var hypo = 1 / (1 + exp(-arg));
          err += (hypo - Y(s)) * X(s, i);
		}
		W(i) = Wcurr(i) - alpha * err;
      }
	}
	execTimes(trial) = getCurrentTime() - startTime;
	if (output) {
      writeln(W);
	}
  }
  printResults(execTimes);
}
