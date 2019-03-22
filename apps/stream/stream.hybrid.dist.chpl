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
config const output = 0: int;
config const alpha = 3.0: real(32);
config param verbose = false;

////////////////////////////////////////////////////////////////////////////////
/// Global Arrays
////////////////////////////////////////////////////////////////////////////////
// For now, these arrays are global so the arrays can be seen from CUDAWrapper
// TODO: Explore the possiblity of declaring the arrays and CUDAWrapper
//       in the main proc (e.g., by using lambdas)
var D: domain(1) dmapped Block(boundingBox = {1..n}) = {1..n};
var A: [D] real(32);
var B: [D] real(32);
var C: [D] real(32);

////////////////////////////////////////////////////////////////////////////////
/// C Interoperability
////////////////////////////////////////////////////////////////////////////////
extern proc streamCUDA(A: [] real(32), B: [] real(32), C: [] real(32), alpha: real(32), lo: int, hi: int, N: int);

// CUDAWrapper is called from GPUIterator
// to invoke a specific CUDA program (using C interoperability)
proc CUDAWrapper(lo: int, hi: int, N: int) {
    if (verbose) {
	writeln("In CUDAWrapper(), launching the CUDA kernel with a range of ", lo, "..", hi, " (Size: ", N, ")");
    }
    ref lA = A.localSlice(lo .. hi);
    ref lB = B.localSlice(lo .. hi);
    ref lC = C.localSlice(lo .. hi);
    streamCUDA(lA, lB, lC, alpha, 0, hi-lo, N);
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
    
    writeln("Stream: CPU/GPU Execution (using GPUIterator)");   
    writeln("Size: ", n);
    writeln("CPU ratio: ", CPUratio);
    writeln("alpha: ", alpha);
    writeln("nTrials: ", numTrials);
    writeln("output: ", output);

    var execTimes: [1..numTrials] real;
    for trial in 1..numTrials {	
	for i in 1..n {
	    B(i) = i: real(32);
	    C(i) = 2*i: real(32);
	}
	
	const startTime = getCurrentTime();
	forall i in GPU(D, CUDAWrapper, CPUratio) {
	    A(i) = B(i) + alpha * C(i);
	}
	execTimes(trial) = getCurrentTime() - startTime;
	if (output) {
	    writeln(A);
	}
    }
    printResults(execTimes);
}
