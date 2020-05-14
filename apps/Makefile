# Chapel
CHPLFLAGS=--fast -sverbose
CHPLMODULE=../../src
GPUAPIFLAGS=-sdebugGPUAPI -sdebugGPUIterator $(CHPLMODULE)/GPUAPI.h GPUAPI.o

# CUDA
CUDA_HOME?=/usr/local/cuda
CUDA_SM?=sm_70
CUDALIBSFLAGS=-L$(CUDA_HOME)/lib64 -lcudart -lcuda -lcublas
NVCCFLAGS=-O3 -arch $(CUDA_SM) -std=c++11 --extended-lambda -I$(CHPLMODULE)
$(info CUDA_HOME is $(CUDA_HOME))
$(info CUDA_SM is $(CUDA_SM))

# ROCM
ROCM_HOME?=/opt/rocm
HIP_HOME=$(ROCM_HOME)/hip
HIPLIBSFLAGS=-L$(ROCM_HOME)/lib -lhip_hcc
$(info ROCM_HOME is $(ROCM_HOME))

# For OpenCL (MacOS)
OCLLIBSFLAGS=-framework OpenCL
OCLFLAGS=-framework OpenCL

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
	chpl -M $(CHPLMODULE) $(CHPLFLAGS) $(TARGET).h $(TARGET).o $(GPUAPIFLAGS) $(TARGET).hybrid.dist.chpl $(CUDALIBSFLAGS)

.PHONY: cudahybrid.dist.lowmid
cudahybrid.dist.lowmid: GPUAPI.o $(TARGET).kernel.cu $(TARGET).hybrid.dist.lowmid.chpl
	nvcc $(NVCCFLAGS) -c $(TARGET).kernel.cu -o $(TARGET).kernel.o
	chpl -M $(CHPLMODULE) $(CHPLFLAGS) $(TARGET).kernel.h $(TARGET).kernel.o $(GPUAPIFLAGS) $(TARGET).hybrid.dist.lowmid.chpl $(CUDALIBSFLAGS) -o $(TARGET).hybrid.dist.lowmid

.PHONY: cudahybrid.dist.mid
cudahybrid.dist.mid: GPUAPI.o $(TARGET).kernel.cu $(TARGET).hybrid.dist.mid.chpl
	nvcc $(NVCCFLAGS) -c $(TARGET).kernel.cu -o $(TARGET).kernel.o
	chpl -M $(CHPLMODULE) $(CHPLFLAGS) $(TARGET).kernel.h $(TARGET).kernel.o $(GPUAPIFLAGS) $(TARGET).hybrid.dist.mid.chpl $(CUDALIBSFLAGS) -o $(TARGET).hybrid.dist.mid

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
	rm -f $(TARGET).baseline $(TARGET).gpu $(TARGET).hybrid $(TARGET).hybrid.dist $(TARGET).hybrid.dist.lowmid $(TARGET).hybrid.dist.mid $(TARGET).o GPUAPI.o *_real
