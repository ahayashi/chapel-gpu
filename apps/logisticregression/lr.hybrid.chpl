use Time;

////////////////////////////////////////////////////////////////////////////////
/// GPUIterator
////////////////////////////////////////////////////////////////////////////////
use GPUIterator;

////////////////////////////////////////////////////////////////////////////////
/// Runtime Options
////////////////////////////////////////////////////////////////////////////////
config const nFeatures = 32: int;
config const nSamples = 32: int;
config const nIters = 32: int;
config const CPUratio1 = 0: int;
config const CPUratio2 = 0: int;
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

// CUDAWrapper is called from GPUIterator
// to invoke a specific CUDA program (using C interoperability)
proc CUDAWrapper1(lo: int, hi: int, N: int) {
    if (verbose) {
	writeln("In CUDAWrapper1(), launching the CUDA kernel with a range of ", lo, "..", hi, " (Size: ", N, ")");
    }
    lrCUDA1(W, Wcurr, lo, hi, N);
}

proc CUDAWrapper2(lo: int, hi: int, N: int) {
    if (verbose) {
	writeln("In CUDAWrapper2(), launching the CUDA kernel with a range of ", lo, "..", hi, " (Size: ", N, ")");
    }
    lrCUDA2(X, Y, W, Wcurr, alpha, nSamples, nFeatures, lo, hi, N);
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

////////////////////////////////////////////////////////////////////////////////
/// Chapel main
////////////////////////////////////////////////////////////////////////////////
proc main() {
    // Assuming there is one locale
    // having CPU and GPU sublocales (CHPL_LOCAL_MODEL=gpu)
    const numSublocs = Locales[0].getChildCount();
    writeln("Locales[0] info: ");
    for sublocID in 0..#numSublocs {
	const subloc = Locales[0].getChild(sublocID);
	writeln("\t Subloc: ", sublocID);
	writeln("\t Name: ", subloc);
	writeln("\t maxTaskPar: ", subloc.maxTaskPar);
    }

    writeln("Logistic Regression: CPU/GPU Execution (using GPUIterator)");
    writeln("nSamples :", nSamples, " nFeatures :",  nFeatures);
    writeln("nTrials: ", numTrials);
    writeln("output: ", output);

    var execTimes: [1..numTrials] real;
    for trial in 1..numTrials {	
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
	
	const startTime = getCurrentTime();
	for ite in 1..nIters {
	    forall i in GPU(1..nFeatures, CUDAWrapper1, CPUratio1) {
		Wcurr(i) = W(i);
	    }
	    forall i in GPU(1..nFeatures, CUDAWrapper2, CPUratio2) {
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
