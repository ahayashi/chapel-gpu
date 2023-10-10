use Time;
use Math;

////////////////////////////////////////////////////////////////////////////////
/// GPUIterator
////////////////////////////////////////////////////////////////////////////////
use GPUIterator;
use GPUAPI;
use BlockDist;
use CTypes;

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
var D: domain(1) dmapped blockDist(boundingBox = {1..n}) = {1..n};
var rand: [D] real(32);
var put: [D] real(32);
var call: [D] real(32);

////////////////////////////////////////////////////////////////////////////////
/// C Interoperability
////////////////////////////////////////////////////////////////////////////////
extern proc LaunchBS(drand: c_ptr(void), dput: c_ptr(void), dcall: c_ptr(void), N: c_size_t);

// CUDAWrapper is called from GPUIterator
// to invoke a specific CUDA program (using C interoperability)
proc CUDAWrapper(lo: int, hi: int, N: int) {
  if (verbose) {
	writeln("In CUDAWrapper(), launching the CUDA kernel with a range of ", lo, "..", hi, " (Size: ", N, ")");
  }
  ref lrand = rand.localSlice(lo .. hi);
  ref lput = put.localSlice(lo .. hi);
  ref lcall = call.localSlice(lo .. hi);
  if (verbose) { ProfilerStart(); }
  var drand = new GPUArray(lrand);
  var dput = new GPUArray(lput);
  var dcall = new GPUArray(lcall);
  toDevice(drand);
  LaunchBS(drand.dPtr(), dput.dPtr(), dcall.dPtr(), N:c_size_t);
  DeviceSynchronize();
  fromDevice(dput, dcall);
  if (verbose) { ProfilerStop(); }
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
  writeln("BlackScholes: CPU/GPU Execution (using GPUIterator)");
  writeln("Size: ", n);
  writeln("CPU ratio: ", CPUratio);
  writeln("nGPUs: ", nGPUs);
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
	forall i in D {
      rand(i) = (i: real(32) / n): real(32);
	}

	const startTime = timeSinceEpoch().totalSeconds();
	forall i in GPU(D, CUDAWrapper, CPUratio)  {
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
      var y = one - oneBySqrt2pi * exp(-X * X / two) * t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));
      if (X  < zero) {
		phiD1 = one - y;
      } else {
		phiD1 = y;
      }
      // phiD2 = phi(d2)
      X = d2;
      absX = abs(X);
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
      absX = abs(X);
      t = one / (one + temp4 * absX);
      y = one - oneBySqrt2pi * exp(-X * X / two) * t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));
      if (X  < zero) {
		phiD1 = one - y;
      } else {
		phiD1 = y;
      }

      // phiD2 = phi(-d2);
      X = -d2;
      absX = abs(X);
      t = one / (one + temp4 * absX);
      y = one - oneBySqrt2pi * exp(-X * X / two) * t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));
      if (X  < zero) {
		phiD2 = one - y;
      } else {
		phiD2 = y;
      }

      put(i) = KexpMinusRT * phiD2 - S * phiD1;
	}
	execTimes(trial) = timeSinceEpoch().totalSeconds() - startTime;
	if (output) {
      writeln("call: ", call);
      writeln("");
      writeln("put: ", put);
	}
  }
  printResults(execTimes);
}
