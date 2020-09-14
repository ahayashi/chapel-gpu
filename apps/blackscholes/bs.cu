#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#define CUDA_ERROR_CHECK
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
}
#endif

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

extern "C" {
    void bsCUDA(float* rand, float *put, float *call, int start, int end, int GPUN) {
        float *drand, *dput, *dcall;

        if (GPUN > 0) {
            assert(end - start + 1 == GPUN);
#ifdef VERBOSE
            printf("In vcCUDA\n");
            printf("\t GPUN: %d\n", GPUN);
            printf("\t range: %d..%d\n", start, end);
#endif
            CudaSafeCall(cudaMalloc(&drand, sizeof(float) * GPUN));
            CudaSafeCall(cudaMalloc(&dput, sizeof(float) * GPUN));
            CudaSafeCall(cudaMalloc(&dcall, sizeof(float) * GPUN));
            CudaSafeCall(cudaMemcpy(drand, rand + start, sizeof(float) * GPUN, cudaMemcpyHostToDevice));

            bs<<<ceil(((float)GPUN)/1024), 1024>>>(drand, dput, dcall, GPUN);
            CudaCheckError();
            CudaSafeCall(cudaDeviceSynchronize());
            CudaSafeCall(cudaMemcpy(put + start, dput, sizeof(float) * GPUN, cudaMemcpyDeviceToHost));
            CudaSafeCall(cudaMemcpy(call + start, dcall, sizeof(float) * GPUN, cudaMemcpyDeviceToHost));

            CudaSafeCall(cudaFree(drand));
            CudaSafeCall(cudaFree(dput));
            CudaSafeCall(cudaFree(dcall));
        }
    }
}
