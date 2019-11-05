# Flags for the Chapel compiler
CHPLFLAGS=--fast
CHPLMODULE=../../src

# For CUDA
CUDALIBSFLAGS=-L/usr/local/cuda/lib64 -lcudart -lcuda
#CUDALIBSFLAGS=-L/opt/apps/software/Compiler/GCC/6.4.0/CUDA/8.0.61/lib -lcudart -lcublas
NVCCFLAGS=-O3 -arch sm_37 -std=c++11

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
cudahybrid: $(TARGET).o $(TARGET).hybrid.chpl
	chpl -M $(CHPLMODULE) $(CHPLFLAGS) $(TARGET).o $(TARGET).hybrid.chpl $(CUDALIBSFLAGS)

.PHONY: cudahybrid.dist
cudahybrid.dist: $(TARGET).o $(TARGET).hybrid.dist.chpl
	chpl -M $(CHPLMODULE) $(CHPLFLAGS) $(TARGET).o $(TARGET).hybrid.dist.chpl $(CUDALIBSFLAGS)

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
	rm -f $(TARGET).baseline $(TARGET).gpu $(TARGET).hybrid $(TARGET).hybrid.dist $(TARGET).o *_real
