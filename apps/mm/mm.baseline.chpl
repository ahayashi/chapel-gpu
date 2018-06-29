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
var A: [1..n, 1..n] real(32);
var B: [1..n, 1..n] real(32);
var C: [1..n, 1..n] real(32);

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

////////////////////////////////////////////////////////////////////////////////
/// Chapel main
////////////////////////////////////////////////////////////////////////////////
proc main() {
    writeln("Matrix Multiplication: Baseline");
    writeln("Size: ", n, "x", n);
    writeln("nTrials: ", numTrials);
    writeln("output: ", output);

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
	forall i in 1..n {
	    forall j in 1..n {
		var sum: real(32) = C(i, j);
		for k in 1..n {
		    sum += A(i, k) * B(k, j);
		}
		C(i, j) = sum;
	    }
	}
	execTimes(trial) = getCurrentTime() - startTime;
	if (output) {
	    writeln(C);
	}
    }
    printResults(execTimes);
}
