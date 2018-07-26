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
extern proc bsCUDA(rand: [] real(32), put: [] real(32), call: [] real(32), lo: int, hi, N: int);

// CUDAWrapper is called from GPUIterator
// to invoke a specific CUDA program (using C interoperability)
proc CUDAWrapper(lo: int, hi: int, N: int) {
    if (verbose) {
	writeln("In CUDAWrapper(), launching the CUDA kernel with a range of ", lo, "..", hi, " (Size: ", N, ")");
    }
    bsCUDA(rand, put, call, lo, hi, N);
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

    writeln("BlackScholes: CPU/GPU Execution (using GPUIterator)");
    writeln("Size: ", n);
    writeln("nTrials: ", numTrials);
    writeln("output: ", output);

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
	
	const startTime = getCurrentTime();
	forall i in GPU(1..n, CUDAWrapper, CPUratio)  {
	    var c1 = 0.319381530: real(32);
	    var c2 = -0.356563782: real(32);
	    var c3 = 1.781477937: real(32);
	    var c4 = -1.821255978: real(32);
	    var c5 = 1.330274429: real(32);
		    
	    var zero = 0.0: real(32);
	    var one = 1.0: real(32);
	    var two = 2.0: real(32);
	    var temp4 = 0.2316419: real(32);
		    
	    var oneBySqrt2pi = 0.398942280: real(32);
		    
	    var inRand = rand(i);
		    
	    var S = S_LOWER_LIMIT * inRand + S_UPPER_LIMIT * (1.0 - inRand);
	    var K = K_LOWER_LIMIT * inRand + K_UPPER_LIMIT * (1.0 - inRand);
	    var T = T_LOWER_LIMIT * inRand + T_UPPER_LIMIT * (1.0 - inRand);
	    var R = R_LOWER_LIMIT * inRand + R_UPPER_LIMIT * (1.0 - inRand);
	    var sigmaVal = SIGMA_LOWER_LIMIT * inRand + SIGMA_UPPER_LIMIT * (1.0 - inRand);
		    
	    var sigmaSqrtT = sigmaVal * sqrt(T);
		    
	    var d1 = (log(S / K) + (R + sigmaVal * sigmaVal / two) * T) / sigmaSqrtT;
	    var d2 = d1 - sigmaSqrtT;
	    
	    var KexpMinusRT = K * exp(-R * T);

	    var phiD1, phiD2: real(32);
		    
	    // phiD1 = phi(d1)
	    var X = d1;
	    var absX = abs(X);
	    var t = one / (one + temp4 * absX);	
	    var y = one - oneBySqrt2pi * Math.exp(-X * X / two) * t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));
	    if (X  < zero) {
		phiD1 = one - y;
	    } else {
		phiD1 = y;
	    }
	    // phiD2 = phi(d2)
	    X = d2;
	    absX = Math.abs(X);
	    t = one / (one + temp4 * absX);	
	    y = one - oneBySqrt2pi * exp(-X * X / two) * t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));
	    if (X  < zero) {
		phiD2 = one - y;
	    } else {
		phiD2 = y;
	    }
		    
	    call(i) = S * phiD1 - KexpMinusRT * phiD2;
		    
	    // phiD1 = phi(-d1);
	    X = -d1;
	    absX = Math.abs(X);
	    t = one / (one + temp4 * absX);	
	    y = one - oneBySqrt2pi * exp(-X * X / two) * t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));
	    if (X  < zero) {
		phiD1 = one - y;
	    } else {
		phiD1 = y;
	    }
	    
	    // phiD2 = phi(-d2);
	    X = -d2;
	    absX = Math.abs(X);
	    t = one / (one + temp4 * absX);	
	    y = one - oneBySqrt2pi * exp(-X * X / two) * t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));
	    if (X  < zero) {
		phiD2 = one - y;
	    } else {
		phiD2 = y;
	    }
		    
	    put(i) = KexpMinusRT * phiD2 - S * phiD1;		
	}	    
	execTimes(trial) = getCurrentTime() - startTime;
	if (output) {
	    writeln(call);
	    writeln(put);
	}
    }
    printResults(execTimes);
}
