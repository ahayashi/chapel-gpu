# Flags for the Chapel compiler
CHPLFLAGS=--fast
CHPLMODULE=../../src

# For CUDA
#CUDALIBSFLAGS=-L/usr/local/cuda/lib64 -lcudart -lcuda
CUDALIBSFLAGS=-L/opt/apps/software/Compiler/GCC/6.4.0/CUDA/8.0.61/lib -lcudart -lcublas
NVCCFLAGS=-O3 -arch sm_37 -std=c++11

# For OpenCL
OPCLLIBSFLAGS=-framework OpenCL
OCLFLAGS=-framework OpenCL

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
	chpl $(CHPLFLAGS) $(TARGET).opencl.o $(TARGET).gpu.chpl --ldflags "$(OCLLIBSFLAGS)"

.PHONY: openclhybrid
oclhybrid: $(TARGET).opencl.o $(TARGET).hybrid.chpl
	chpl -M $(CHPLMODULE) $(CHPLFLAGS) $(TARGET).opencl.o $(TARGET).hybrid.chpl --ldflags "$(OCLLIBSFLAGS)"

.PHONY: clean
clean:
	rm -f $(TARGET).baseline $(TARGET).gpu $(TARGET).hybrid $(TARGET).hybrid.dist $(TARGET).o *_real
