TARGET=stream
CHPLFLAGS=--fast -sverbose
LIBSFLAGS=-L/usr/local/cuda/lib64 -lcudart -lcuda
NVCCFLAGS=-O3 -arch sm_60 -std=c++11

all: baseline gpu hybrid hybrid.dist

$(TARGET).o: $(TARGET).cu
	nvcc $(NVCCFLAGS) -c $^

.PHONY: baseline
baseline: $(TARGET).baseline.chpl
	CHPL_LOCALE_MODEL=flat chpl $(CHPLFLAGS) $(TARGET).baseline.chpl

.PHONY: gpu
gpu: $(TARGET).o $(TARGET).gpu.chpl
	CHPL_LOCALE_MODEL=flat chpl $(CHPLFLAGS) $(TARGET).o $(TARGET).gpu.chpl $(LIBSFLAGS)

.PHONY: hybrid
hybrid: $(TARGET).o $(TARGET).hybrid.chpl
	CHPL_LOCALE_MODEL=gpu chpl -M ../../modules $(CHPLFLAGS) $(TARGET).o $(TARGET).hybrid.chpl $(LIBSFLAGS)

.PHONY: hybrid.dist
hybrid.dist: $(TARGET).o $(TARGET).hybrid.dist.chpl
	CHPL_LOCALE_MODEL=gpu chpl -M ../../modules $(CHPLFLAGS) $(TARGET).o $(TARGET).hybrid.dist.chpl $(LIBSFLAGS)

.PHONY: clean
clean:
	rm -f $(TARGET).baseline $(TARGET).gpu $(TARGET).hybrid $(TARGET).hybrid.dist $(TARGET).o *_real
