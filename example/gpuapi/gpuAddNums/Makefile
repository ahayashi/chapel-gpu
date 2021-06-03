CHPL_GPU_MODULES=-M $(CHPL_GPU_HOME)/modules $(CHPL_GPU_HOME)/include/GPUAPI.h
CHPL_FLAGS=--fast $(CHPL_GPU_MODULES)
CHPL_GPU_LIB_PATH=-L$(CHPL_GPU_HOME)/lib -L$(CHPL_GPU_HOME)/lib64
CUDA_LIBS=$(CHPL_GPU_LIB_PATH) -lGPUAPICUDA_static -L$(CUDA_ROOT_DIR)/lib -lcuda -lcudart $(CUBLAS_LIB)

all: cuda

cuda: gpuAddNums.chpl
	CHPL_LOCALE_MODEL=gpu chpl $(CHPL_GPU_MODULES) --llvm --ccflags --cuda-gpu-arch=sm_60 --savec=tmp gpuAddNums.chpl $(CUDA_LIBS) -o gpuAddNums.$@

.PHONY: clean
clean:
	rm -rf gpuAddNums.cuda tmp
