TARGET=bs
CHPLFLAGS=--fast -sverbose
LIBSFLAGS=-L/usr/local/cuda/lib64 -lcudart -lcuda
NVCCFLAGS=-O3 -arch sm_60

# OpenCL
OCLLIBSFLAGS=-framework OpenCL
OCLFLAGS=-framework OpenCL

all: baseline gpu hybrid

$(TARGET).o: $(TARGET).cu
	nvcc $(NVCCFLAGS) -c $^

$(TARGET).opencl.o: $(TARGET).opencl.c
	gcc -O3 -Wall $(OCLFLAGS) -c $^

.PHONY: baseline
baseline: $(TARGET).baseline.chpl
	CHPL_LOCALE_MODEL=flat chpl $(CHPLFLAGS) $(TARGET).baseline.chpl

.PHONY: gpu
gpu: $(TARGET).o $(TARGET).gpu.chpl
	CHPL_LOCALE_MODEL=flat chpl $(CHPLFLAGS) $(TARGET).o $(TARGET).gpu.chpl $(LIBSFLAGS)

.PHONY: oclgpu
oclgpu: $(TARGET).opencl.o $(TARGET).gpu.chpl
	CHPL_LOCALE_MODEL=flat chpl $(CHPLFLAGS) $(TARGET).opencl.o $(TARGET).gpu.chpl --ldflags "$(OCLLIBSFLAGS)"

.PHONY: hybrid
hybrid: $(TARGET).o $(TARGET).hybrid.chpl
	CHPL_LOCALE_MODEL=gpu chpl -M ../../modules $(CHPLFLAGS) $(TARGET).o $(TARGET).hybrid.chpl $(LIBSFLAGS)

.PHONY: oclhybrid
oclhybrid: $(TARGET).opencl.o $(TARGET).hybrid.chpl
	CHPL_LOCALE_MODEL=gpu chpl -M ../../modules $(CHPLFLAGS) $(TARGET).opencl.o $(TARGET).hybrid.chpl --ldflags "$(OCLLIBSFLAGS)"


.PHONY: clean
clean:
	rm -f $(TARGET).baseline $(TARGET).gpu $(TARGET).hybrid $(TARGET).o
