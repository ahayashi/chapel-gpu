#define S_LOWER_LIMIT 10.0f
#define S_UPPER_LIMIT 100.0f
#define K_LOWER_LIMIT 10.0f
#define K_UPPER_LIMIT 100.0f
#define T_LOWER_LIMIT 1.0f
#define T_UPPER_LIMIT 10.0f
#define R_LOWER_LIMIT 0.01f
#define R_UPPER_LIMIT 0.05f
#define SIGMA_LOWER_LIMIT 0.01f
#define SIGMA_UPPER_LIMIT 0.10f

#ifndef USE_LAMBDA
__global__ void bs(float *drand, float *dput, float *dcall, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
	float c1 = 0.319381530f;
	float c2 = -0.356563782f;
	float c3 = 1.781477937f;
	float c4 = -1.821255978f;
	float c5 = 1.330274429f;
		    
	float zero = 0.0f;
	float one = 1.0f;
	float two = 2.0f;
	float temp4 = 0.2316419f;
		    
	float oneBySqrt2pi = 0.398942280f;
		    
	float d1, d2;
	float phiD1, phiD2;
	float sigmaSqrtT;
	float KexpMinusRT;
		    
	float inRand;		    
		    
	inRand = drand[id];
		    
	float S = S_LOWER_LIMIT * inRand + S_UPPER_LIMIT * (1.0f - inRand);
	float K = K_LOWER_LIMIT * inRand + K_UPPER_LIMIT * (1.0f - inRand);
	float T = T_LOWER_LIMIT * inRand + T_UPPER_LIMIT * (1.0f - inRand);
	float R = R_LOWER_LIMIT * inRand + R_UPPER_LIMIT * (1.0f - inRand);
	float sigmaVal = SIGMA_LOWER_LIMIT * inRand + SIGMA_UPPER_LIMIT * (1.0f - inRand);
		    
	sigmaSqrtT = sigmaVal * (float)sqrt(T);
		    
	d1 = ((float)log(S / K) + (R + sigmaVal * sigmaVal / two) * T) / sigmaSqrtT;
	d2 = d1 - sigmaSqrtT;
		    
	KexpMinusRT = K * (float)exp(-R * T);
		    
	// phiD1 = phi(d1)
	float X = d1;
	float absX = (float)abs(X);
	float t = one / (one + temp4 * absX);	
	float y = one - oneBySqrt2pi * (float)exp(-X * X / two) * t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));
	phiD1 = (X < zero) ? (one - y) : y; 
	// phiD2 = phi(d2)
	X = d2;
	absX = abs(X);
	t = one / (one + temp4 * absX);	
	y = one - oneBySqrt2pi * (float)exp(-X * X / two) * t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));
	phiD2 = (X < zero) ? (one - y) : y; 
		    
	dcall[id] = S * phiD1 - KexpMinusRT * phiD2;

	// phiD1 = phi(-d1);
	X = -d1;
	absX = abs(X);
	t = one / (one + temp4 * absX);	
	y = one - oneBySqrt2pi * (float)exp(-X * X / two) * t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));
	phiD1 = (X < zero) ? (one - y) : y; 
		    
	// phiD2 = phi(-d2);
	X = -d2;
	absX = abs(X);
	t = one / (one + temp4 * absX);	
	y = one - oneBySqrt2pi * (float)exp(-X * X / two) * t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));
	phiD2 = (X < zero) ? (one - y) : y; 
		    
	dput[id] = KexpMinusRT * phiD2 - S * phiD1;			
    }
}
#else
#include "lambda.h"
#endif

extern "C" {
#ifndef USE_LAMBDA    
    void LaunchBS(float* drand, float *dput, float *dcall, int N) {
        bs<<<ceil(((float)N)/1024), 1024>>>(drand, dput, dcall, N);
    }
#else
    void LaunchBS(float* drand, float *dput, float *dcall, int N) {    
        call_gpu_functor(N, 1024, NULL, [=] __device__ (int id) {
                float c1 = 0.319381530f;
                float c2 = -0.356563782f;
                float c3 = 1.781477937f;
                float c4 = -1.821255978f;
                float c5 = 1.330274429f;
                
                float zero = 0.0f;
                float one = 1.0f;
                float two = 2.0f;
                float temp4 = 0.2316419f;
                
                float oneBySqrt2pi = 0.398942280f;
                
                float d1, d2;
                float phiD1, phiD2;
                float sigmaSqrtT;
                float KexpMinusRT;
                
                float inRand;		    
                
                inRand = drand[id];
                
                float S = S_LOWER_LIMIT * inRand + S_UPPER_LIMIT * (1.0f - inRand);
                float K = K_LOWER_LIMIT * inRand + K_UPPER_LIMIT * (1.0f - inRand);
                float T = T_LOWER_LIMIT * inRand + T_UPPER_LIMIT * (1.0f - inRand);
                float R = R_LOWER_LIMIT * inRand + R_UPPER_LIMIT * (1.0f - inRand);
                float sigmaVal = SIGMA_LOWER_LIMIT * inRand + SIGMA_UPPER_LIMIT * (1.0f - inRand);
                
                sigmaSqrtT = sigmaVal * (float)sqrt(T);
                
                d1 = ((float)log(S / K) + (R + sigmaVal * sigmaVal / two) * T) / sigmaSqrtT;
                d2 = d1 - sigmaSqrtT;
                
                KexpMinusRT = K * (float)exp(-R * T);
                
                // phiD1 = phi(d1)
                float X = d1;
                float absX = (float)abs(X);
                float t = one / (one + temp4 * absX);	
                float y = one - oneBySqrt2pi * (float)exp(-X * X / two) * t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));
                phiD1 = (X < zero) ? (one - y) : y; 
                // phiD2 = phi(d2)
                X = d2;
                absX = abs(X);
                t = one / (one + temp4 * absX);	
                y = one - oneBySqrt2pi * (float)exp(-X * X / two) * t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));
                phiD2 = (X < zero) ? (one - y) : y; 
                
                dcall[id] = S * phiD1 - KexpMinusRT * phiD2;
                
                // phiD1 = phi(-d1);
                X = -d1;
                absX = abs(X);
                t = one / (one + temp4 * absX);	
                y = one - oneBySqrt2pi * (float)exp(-X * X / two) * t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));
                phiD1 = (X < zero) ? (one - y) : y; 
                
                // phiD2 = phi(-d2);
                X = -d2;
                absX = abs(X);
                t = one / (one + temp4 * absX);	
                y = one - oneBySqrt2pi * (float)exp(-X * X / two) * t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5))));
                phiD2 = (X < zero) ? (one - y) : y; 
                
                dput[id] = KexpMinusRT * phiD2 - S * phiD1;			
            });
}
#endif    
}
