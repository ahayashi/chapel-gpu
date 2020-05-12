# Flags for the Chapel compiler
CHPLFLAGS=--fast -sverbose
CHPLMODULE=../../src
GPUAPIFLAGS=-sdebugGPUAPI -sdebugGPUIterator $(CHPLMODULE)/GPUAPI.h GPUAPI.o

# For CUDA
CUDALIBSFLAGS=-L$(CUDA_HOME)/lib64 -lcudart -lcuda
#CUDALIBSFLAGS=-L/sw/summit/cuda/10.1.168/lib64 -lcudart -lcuda
#CUDALIBSFLAGS=-L/opt/apps/software/Compiler/GCC/6.4.0/CUDA/8.0.61/lib -lcudart -lcublas
NVCCFLAGS=-O3 -arch sm_37 -std=c++11 --extended-lambda -I$(CHPLMODULE)

# For OpenCL
OCLLIBSFLAGS=-framework OpenCL
OCLFLAGS=-framework OpenCL

# For HIP
ROCM_HOME=/opt/rocm
HIP_HOME=$(ROCM_HOME)/hip
HIPLIBSFLAGS=-L$(ROCM_HOME)/lib -lhip_hcc

all: baseline cudagpu cudahybrid cudahybrid.dist

$(TARGET).o: $(TARGET).cu
	nvcc $(NVCCFLAGS) -c $^

GPUAPI.o: $(CHPLMODULE)/GPUAPI.cu
	nvcc $(NVCCFLAGS) -c $^

$(TARGET).opencl.o: $(TARGET).opencl.c
	gcc -O3 -Wall $(OCLFLAGS) -c $^

.PHONY: baseline
baseline: $(TARGET).baseline.chpl
	chpl $(CHPLFLAGS) $(TARGET).baseline.chpl

.PHONY: blas
blas:
	chpl $(CHPLFLAGS) $(TARGET).blas.chpl

.PHONY: cudagpu
cudagpu: $(TARGET).o $(TARGET).gpu.chpl
	chpl $(CHPLFLAGS) $(TARGET).o $(TARGET).gpu.chpl $(CUDALIBSFLAGS)

.PHONY: cudahybrid
cudahybrid: GPUAPI.o $(TARGET).o $(TARGET).hybrid.chpl
	chpl -M $(CHPLMODULE) $(CHPLFLAGS) $(TARGET).o $(GPUAPIFLAGS) $(TARGET).hybrid.chpl $(CUDALIBSFLAGS)

.PHONY: cudahybrid.dist
cudahybrid.dist: GPUAPI.o $(TARGET).o $(TARGET).hybrid.dist.chpl
	chpl -M $(CHPLMODULE) $(CHPLFLAGS) $(TARGET).o $(GPUAPIFLAGS) $(TARGET).hybrid.dist.chpl $(CUDALIBSFLAGS)

.PHONY: cudahybrid.dist.explicit
cudahybrid.dist.explicit: GPUAPI.o $(TARGET).kernel.cu $(TARGET).hybrid.dist.explicit.chpl
	nvcc $(NVCCFLAGS) -DUSE_LAMBDA -c $(TARGET).kernel.cu -o $(TARGET).lambda.o
	nvcc $(NVCCFLAGS)              -c $(TARGET).kernel.cu -o $(TARGET).kernel.o
	chpl -M $(CHPLMODULE) $(CHPLFLAGS) $(TARGET).kernel.h $(TARGET).kernel.o $(GPUAPIFLAGS) $(TARGET).hybrid.dist.explicit.chpl $(CUDALIBSFLAGS) -o $(TARGET).hybrid.dist.explicit.kernel
	chpl -M $(CHPLMODULE) $(CHPLFLAGS) $(TARGET).kernel.h $(TARGET).lambda.o $(GPUAPIFLAGS) $(TARGET).hybrid.dist.explicit.chpl $(CUDALIBSFLAGS) -o $(TARGET).hybrid.dist.explicit.lambda

.PHONY: cudahybrid.dist.explicit.mid
cudahybrid.dist.explicit.mid: GPUAPI.o $(TARGET).kernel.cu $(TARGET).hybrid.dist.explicit.mid.chpl
	nvcc $(NVCCFLAGS) -DUSE_LAMBDA -c $(TARGET).kernel.cu -o $(TARGET).lambda.o
	nvcc $(NVCCFLAGS)              -c $(TARGET).kernel.cu -o $(TARGET).kernel.o
	chpl -M $(CHPLMODULE) $(CHPLFLAGS) $(TARGET).kernel.h $(TARGET).kernel.o $(GPUAPIFLAGS) $(TARGET).hybrid.dist.explicit.mid.chpl $(CUDALIBSFLAGS) -o $(TARGET).hybrid.dist.explicit.mid.kernel
	chpl -M $(CHPLMODULE) $(CHPLFLAGS) $(TARGET).kernel.h $(TARGET).lambda.o $(GPUAPIFLAGS) $(TARGET).hybrid.dist.explicit.mid.chpl $(CUDALIBSFLAGS) -o $(TARGET).hybrid.dist.explicit.mid.lambda


.PHONY: openclgpu
oclgpu: $(TARGET).opencl.o $(TARGET).gpu.chpl
	chpl $(CHPLFLAGS) $(TARGET).opencl.o $(TARGET).gpu.chpl --ldflags $(OCLLIBSFLAGS)

.PHONY: openclhybrid
oclhybrid: $(TARGET).opencl.o $(TARGET).hybrid.chpl
	chpl -M $(CHPLMODULE) $(CHPLFLAGS) $(TARGET).opencl.o $(TARGET).hybrid.chpl --ldflags $(OCLLIBSFLAGS)

.PHONY: hipgpu
hipgpu: $(TARGET).cu $(TARGET).gpu.chpl
	$(HIP_HOME)/bin/hipify-perl $(TARGET).cu > $(TARGET).hip.cpp
	$(HIP_HOME)/bin/hipcc -O3 -Wall -fPIC -c $(TARGET).hip.cpp -fno-gpu-rdc
	chpl $(CHPLFLAGS) $(TARGET).hip.o $(TARGET).gpu.chpl --ldflags $(HIPLIBSFLAGS)

.PHONY: hiphybrid
hiphybrid: $(TARGET).cu $(TARGET).hybrid.chpl
	$(HIP_HOME)/bin/hipify-perl $(TARGET).cu > $(TARGET).hip.cpp
	$(HIP_HOME)/bin/hipcc -O3 -Wall -fPIC -c $(TARGET).hip.cpp -fno-gpu-rdc
	chpl -M $(CHPLMODULE) $(CHPLFLAGS) $(TARGET).hip.o $(TARGET).hybrid.chpl --ldflags $(HIPLIBSFLAGS)

.PHONY: clean
clean:
	rm -f $(TARGET).baseline $(TARGET).gpu $(TARGET).hybrid $(TARGET).hybrid.dist $(TARGET).hybrid.dist.explicit.kernel $(TARGET).hybrid.dist.explicit.lambda $(TARGET).o *_real
