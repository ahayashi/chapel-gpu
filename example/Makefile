CHPL_GPU_MODULES=-M $(CHPL_GPU_HOME)/modules $(CHPL_GPU_HOME)/include/GPUAPI.h
CHPL_FLAGS=--fast $(CHPL_GPU_MODULES) -suseGPU=true

# CUDA
CUDA_LIBS=-L$(CHPL_GPU_HOME)/lib -lGPUAPICUDA_static -L./build -lvc.cuda -L$(CUDA_ROOT_DIR)/lib -lcudart
# HIP
HIP_LIBS=-L$(CHPL_GPU_HOME)/lib -lGPUAPIHIP_static -L./build -lvc.hip -L$(HIP_ROOT_DIR)/lib -lhip_hcc

# OpenCL
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	OPENCL_LIBS=-L$(CHPL_GPU_HOME)/lib -lGPUAPIOPENCL_static -L./build -lvc.opencl --ldflags '-framework OpenCL'
else
	OPENCL_LIBS=-L$(CHPL_GPU_HOME)/lib -lGPUAPIOPENCL_static -L./build -lvc.opencl -L$(subst libOpenCL.so,,$(OpenCL_LIBRARIES)) -lOpenCL
endif

all: cuda

gpu: vc.cu
	rm -rf build &&	mkdir build && cd build && cmake .. && make

cuda: gpu vc.chpl
	chpl $(CHPL_FLAGS) vc.h vc.chpl $(CUDA_LIBS) -o vc.cuda

hip: gpu vc.chpl
	chpl $(CHPL_FLAGS) vc.h vc.chpl $(HIP_LIBS) -o vc.hip

opencl: gpu vc.chpl
	chpl $(CHPL_FLAGS) vc.h vc.chpl $(OPENCL_LIBS) -o vc.opencl

