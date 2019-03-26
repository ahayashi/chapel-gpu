use Time;

////////////////////////////////////////////////////////////////////////////////
/// Runtime Options
////////////////////////////////////////////////////////////////////////////////
config const n = 32: int;
config const numTrials = 1: int;
config const output = 0: int;
config const alpha = 3.0: real(32);
config param verbose = false;

////////////////////////////////////////////////////////////////////////////////
/// Global Arrays
////////////////////////////////////////////////////////////////////////////////
// For now, these arrays are global so the arrays can be seen from CUDAWrapper
// TODO: Explore the possiblity of declaring the arrays and CUDAWrapper
//       in the main proc (e.g., by using lambdas)
var A: [1..n] real(32);
var B: [1..n] real(32);
var C: [1..n] real(32);

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
        const numSublocs = loc.getChildCount();
        writeln(loc, " info: ");
        for sublocID in 0..#numSublocs {
            const subloc = loc.getChild(sublocID);
            writeln("\t Subloc: ", sublocID);
            writeln("\t Name: ", subloc);
            writeln("\t maxTaskPar: ", subloc.maxTaskPar);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
/// Chapel main
////////////////////////////////////////////////////////////////////////////////
proc main() {
    writeln("Stream Baseline");
    writeln("Size: ", n);
    writeln("alpha: ", alpha);
    writeln("nTrials: ", numTrials);
    writeln("output: ", output);

    printLocaleInfo();

    var execTimes: [1..numTrials] real;
    for trial in 1..numTrials {
	for i in 1..n {
	    B(i) = i: real(32);
	    C(i) = 2*i: real(32);
	}

	const startTime = getCurrentTime();
	forall i in 1..n {
	    A(i) = B(i) + alpha * C(i);
	}
	execTimes(trial) = getCurrentTime() - startTime;
	if (output) {
	    writeln(A);
	}
    }
    printResults(execTimes);
}
