
//https://github.com/ernestyalumni/CompPhys/blob/master/moreCUDA/CUBLAS/036sgemm.c

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*(ld))+( i ))
#define m 2 									// a - mxk matrix
#define n 4 									// b - kxn matrix
#define k 3 									// c - mxn matrix
int main(void) {
	cudaError_t cudaStat; 					// cudaMalloc status
	cublasStatus_t stat; 			// CUBLAS functions status
	cublasHandle_t handle;						// CUBLAS context
	int i,j;						// i-row index,j-column index
	float* a;							// mxk matrix a on the host
	float* b;							// kxn matrix b on the host
	float* c; 							// mxn matrix c on the host
	a=(float*)malloc(m*k*sizeof(float));		// host memory for a
	b=(float*)malloc(k*n*sizeof(float));		// host memory for b
	c=(float*)malloc(m*n*sizeof(float));		// host memory for c
	// define an mxk matrix a column by column
	int ind=11;										// a:
	for (j=0;j<k;j++){								// 11,17,23,29,35
		for (i=0;i<m;i++){							// 12,18,24,30,36
			a[IDX2C(i,j,m)]=(float)ind++;			// 13,19,25,31,37
		}											// 14,20,26,32,38
	}												// 15,21,27,33,39
													// 16,22,28,34,40
    int co = 0;
    for (i=0;i<m;i++){
        for (j=0;j<k;j++){
            a[co] = co+1;
            co++;
        }
    }
	// print a row by row
    co=0;
	printf("a:\n");
		for (i=0;i<m;i++) {
			for (j=0;j<k;j++){
				//printf("%5.0f",a[IDX2C(i,j,m)]);
				printf("%5.0f",a[co++]);
			}
		printf("\n");
	}
	// define a kxn matrix b column by column
	ind=11;												// b:
	for (j=0;j<n;j++) {									// 11,16,21,26
		for (i=0;i<k;i++) {								// 12,17,22,27
			b[IDX2C(i,j,k)]=(float)ind++;				// 13,18,23,28
		}												// 14,19,24,29
    }												// 15,20,25,30

    co = 0;
    for (i=0;i<k;i++){
        for (j=0;j<n;j++){
            b[co] = co+1;
            co++;
        }
    }
	// print b row by row
    co=0;
	printf("b: \n");
	for(i=0;i<k;i++){
		for (j=0;j<n;j++) {
			printf("%5.0f",b[co++]);
		}
		printf("\n");
	}
	// define an mxn matrix c column by column
    co = 0;
	ind =11;												// c:
	for (j=0;j<n;j++){									// 11,17,23,29
		for (i=0;i<m;i++){								// 12,18,24,30
			//c[IDX2C(i,j,m)]=(float)ind++;				// 13,19,25,31
            c[co++] = 0;
		}												// 14,20,26,32
	}													// 15,21,27,33
														// 16,22,28,34
	// print c row by row
    co=0;
	printf("c:\n");
		for (i=0;i<m;i++) {
			for (j=0;j<n;j++) {
				printf("%5.0f",c[co++]);
			}
			printf("\n");
		}

	// on the device
	float* d_a; 							// d_a - a on the device
	float* d_b; 							// d_b - b on the device
	float* d_c;								// d_c - c on the device
	cudaStat=cudaMalloc((void**)&d_a,m*k*sizeof(*a));	// device
												// memory alloc for a
	cudaStat=cudaMalloc((void**)&d_b,k*n*sizeof(*b)); 	// device
												// memory alloc for b
	cudaStat=cudaMalloc((void**)&d_c,m*n*sizeof(*c));	// device
												// memory alloc for c
	stat = cublasCreate(&handle); 		// initialize CUBLAS context
	// copy matrices from the host to the device
	stat = cublasSetMatrix(m,k,sizeof(*a),a,m,d_a,m); 	// a -> d_a
	stat = cublasSetMatrix(k,n,sizeof(*b),b,k,d_b,k);	// b -> d_b
	stat = cublasSetMatrix(m,n,sizeof(*c),c,m,d_c,m);	// c -> d_c
	float a1=1.0f;											// a1=1
	float bet=0.0f;											// bet=1
	// matrix-matrix multiplication: d_c = a1*d_a*d_b + bet*d_c
	// d_a -mxk matrix, d_b - kxn matrix, d_c -mxn matrix;
	// a1,bet - scalars

    //c with some sort of transposed way
    stat=cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,m,n,k,&a1,d_a,k,d_b,n,&bet,d_c,m);

	stat=cublasGetMatrix(m,n,sizeof(*c),d_c,m,c,m);	// cp d_c -> c
	printf("c after Sgemm :\n");
    co=0;
	for(i=0;i<m;i++){
		for (j=0;j<n;j++){
			printf("%7.0f", c[co++]); 	// print c after Sgemm
		}
		printf("\n");
	}

    //C
	stat=cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&a1,d_b,n,d_a,k,&bet,d_c,n);
    stat=cublasGetMatrix(m,n,sizeof(*c),d_c,m,c,m);	// cp d_c -> c
	printf("c after Sgemm :\n");
    co=0;
	for(i=0;i<m;i++){
		for (j=0;j<n;j++){
			printf("%7.0f", c[co++]); 	// print c after Sgemm
		}
		printf("\n");
	}

	cudaFree(d_a);								// free device memory
	cudaFree(d_b);								// free device memory
	cudaFree(d_c);								// free device memory
	cublasDestroy(handle);					// destroy CUBLAS context
	free(a);									// free host memory
	free(b);									// free host memory
	free(c);									// free host memory
	return EXIT_SUCCESS;
}

// a:
//   11   17   23   29   35
//   12   18   24   30   36
//   13   19   25   31   37
//   14   20   26   32   38
//   15   21   27   33   39
//   16   22   28   34   40
// b:
//   11   16   21   26
//   12   17   22   27
//   13   18   23   28
//   14   19   24   29
//   15   20   25   30
// c:
//   11   17   23   29
//   12   18   24   30
//   13   19   25   31
//   14   20   26   32
//   15   21   27   33
//   16   22   28   34
// c after Sgemm :
//   1566   2147   2728   3309
//   1632   2238   2844   3450  	// c=a1*a*b+bet*c
//   1698   2329   2960   3591
//   1764   2420   3076   3732
//   1830   2511   3192   3873
//   1896   2602   3308   4014


