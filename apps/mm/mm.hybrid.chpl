use Time;

////////////////////////////////////////////////////////////////////////////////
/// GPUIterator
////////////////////////////////////////////////////////////////////////////////
use GPUIterator;

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
var A: [1..n, 1..n] real(32);
var B: [1..n, 1..n] real(32);
var C: [1..n, 1..n] real(32);

////////////////////////////////////////////////////////////////////////////////
/// C Interoperability
////////////////////////////////////////////////////////////////////////////////
extern proc mmCUDA(A: [] real(32), B: [] real(32), C: [] real(32), N:int, lo: int, hi: int, GPUN: int, tiled: int);

// CUDAWrapper is called from GPUIterator
// to invoke a specific CUDA program (using C interoperability)
proc CUDAWrapper(lo: int, hi: int, N: int) {
    if (verbose) {
	writeln("In CUDAWrapper(), launching the CUDA kernel with a range of ", lo, "..", hi, " (Size: ", N, ")");
    }
    mmCUDA(A, B, C, n*n, lo, hi, N, tiled);
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
    
    writeln("Matrix Multiplication: CPU/GPU Execution (using GPUIterator)");
    writeln("Size: ", n, "x", n);
    writeln("CPU ratio: ", CPUratio);
    writeln("nTrials: ", numTrials);
    writeln("tiled: ", tiled);
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
	// TODO: Consider using a 2D iterator
	forall e in GPU(1, n*n, CUDAWrapper, CPUratio) {
	    var i: int = (e - 1) / n + 1;
	    var j: int = (e - 1) % n + 1;
	    var sum: real(32) = C(i, j);
	    for k in 1..n {
		sum += A(i, k) * B(k, j);
	    }
	    C(i, j) = sum;
	}
	execTimes(trial) = getCurrentTime() - startTime;
	if (output) {
	    writeln(C);
	}
    }
    printResults(execTimes);
}
