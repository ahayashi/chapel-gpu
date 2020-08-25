CHPL_GPU_MODULES=-M $(CHPL_GPU_HOME)/modules $(CHPL_GPU_HOME)/include/GPUAPI.h
CHPL_FLAGS=--fast $(CHPL_GPU_MODULES) -suseGPU=true

CUDA_LIBS=-L$(CHPL_GPU_HOME)/lib -lGPUAPICUDA_static -L./build -lvc.cuda -L$(CUDA_ROOT_DIR)/lib -lcudart

all: cuda

gpu: vc.cu
	rm -rf build &&	mkdir build && cd build && cmake .. && make

cuda: gpu vc.chpl
	chpl $(CHPL_FLAGS) vc.h vc.chpl $(CUDA_LIBS)

