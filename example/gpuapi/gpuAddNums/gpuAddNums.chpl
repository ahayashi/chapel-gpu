extern {
  #include <stdio.h>
  #include <cuda.h>
  #include <assert.h>

  #define FATBIN_FILE "tmp/chpl__gpu.fatbin"

  static void checkCudaErrors(CUresult err) {
    assert(err == CUDA_SUCCESS);
  }

  static void launchKernel(void *dA){
    CUdevice    device;
    CUmodule    cudaModule;
    CUcontext   context;
    CUfunction  function;
    int         devCount;

    checkCudaErrors(cuInit(0));
    checkCudaErrors(cuDeviceGetCount(&devCount));
    checkCudaErrors(cuDeviceGet(&device, 0));

    char name[128];
    checkCudaErrors(cuDeviceGetName(name, 128, device));

    checkCudaErrors(cuCtxCreate(&context, 0, device));

    char * buffer = 0;
    long length;
    FILE * f = fopen (FATBIN_FILE, "rb");

    if (f)
    {
      fseek (f, 0, SEEK_END);
      length = ftell (f);
      fseek (f, 0, SEEK_SET);
      buffer = (char* )malloc (length);
      if (buffer)
      {
        fread (buffer, 1, length, f);
      }
      fclose (f);
    }


    checkCudaErrors(cuModuleLoadData(&cudaModule, buffer));

    checkCudaErrors(cuModuleGetFunction(&function, cudaModule, "add_nums"));

    unsigned blockSizeX = 1;
    unsigned blockSizeY = 1;
    unsigned blockSizeZ = 1;
    unsigned gridSizeX  = 1;
    unsigned gridSizeY  = 1;
    unsigned gridSizeZ  = 1;

    void *KernelParams[] = { &dA };

    checkCudaErrors(cuLaunchKernel(function, gridSizeX, gridSizeY, gridSizeZ,
                                   blockSizeX, blockSizeY, blockSizeZ,
                                   0, NULL, KernelParams, NULL));


  }

}

pragma "codegen for GPU"
pragma "always resolve function"
export proc add_nums(dst_ptr: c_ptr(real(64))){
    dst_ptr[0] = dst_ptr[0] + 10;
}

use GPUAPI;
var D = {0..#1};
var A: [D] real(64);
A[0] = 10;
var dA = new GPUArray(A);
dA.toDevice();
launchKernel(dA.dPtr());
dA.fromDevice();
writeln(A);