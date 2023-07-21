CHPL_GPU_MODULES=-M $(CHPL_GPU_HOME)/modules $(CHPL_GPU_HOME)/include/GPUAPI.h
CHPL_FLAGS=--fast $(CHPL_GPU_MODULES) -suseGPU=true --ldflags -no-pie --savec tmp
CHPL_GPU_LIB_PATH=-L$(CHPL_GPU_HOME)/lib -L$(CHPL_GPU_HOME)/lib64
CHPL_GPU_CMAKE_PATH=$(CHPL_GPU_HOME)/share/cmake/ChapelGPUAPI

# CUDA
CUDA_LIBS=$(CHPL_GPU_LIB_PATH) -lGPUAPICUDA_static -L./build -lvc.cuda -L$(CUDA_ROOT_DIR)/lib -lcudart
# HIP
HIP_LIBS=$(CHPL_GPU_LIB_PATH) -lGPUAPIHIP_static -L./build -lvc.hip -L$(HIP_ROOT_DIR)/lib -lamdhip64
# DPC++
DPCPP_LIBS=$(CHPL_GPU_LIB_PATH) -lGPUAPIDPCPP_static $(DBLAS_LIB)
DPCPP_FLAGS=-DTHREADS_PER_BLOCK=256 $(DBLAS_FLAG)

# OpenCL
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	OPENCL_LIBS=$(CHPL_GPU_LIB_PATH) -lGPUAPIOPENCL_static -L./build -lvc.opencl --ldflags '-framework OpenCL'
else
	OPENCL_LIBS=$(CHPL_GPU_LIB_PATH) -lGPUAPIOPENCL_static -L./build -lvc.opencl -L$(subst libOpenCL.so,,$(OpenCL_LIBRARIES)) -lOpenCL
endif

build/libvc.cuda.a: vc.cu
	rm -rf build &&	mkdir build && cd build && cmake .. -DCMAKE_MODULE_PATH=$(CHPL_GPU_CMAKE_PATH) && make

build/libvc.hip.a: vc.cu
	rm -rf build &&	mkdir build && cd build && cmake .. -DCMAKE_MODULE_PATH=$(CHPL_GPU_CMAKE_PATH) && make

build/libvc.dpcpp.a: vc.cu
	rm -rf build &&	mkdir build && cd build && cmake .. -DCMAKE_MODULE_PATH=$(CHPL_GPU_CMAKE_PATH) && make

build/libvc.opencl.a: vc.opencl.c
	rm -rf build &&	mkdir build && cd build && cmake .. -DCMAKE_MODULE_PATH=$(CHPL_GPU_CMAKE_PATH) && make

cuda: vc.chpl vc.h build/libvc.cuda.a
	chpl $(CHPL_FLAGS) vc.chpl vc.h $(CUDA_LIBS) -o vc.cuda

hip: vc.chpl vc.h build/libvc.hip.a
	chpl $(CHPL_FLAGS) vc.chpl vc.h $(HIP_LIBS) -o vc.hip

dpcpp: vc.chpl vc.h build/libvc.dpcpp.a
	rm -rf tmp
	-chpl --savec tmp $(CHPL_FLAGS) vc.chpl
	dpcpp $(DPCPP_FLAGS) build/vc.dp.cpp tmp/vc.tmp.o $(CHPL_HOME)/lib/`$(CHPL_HOME)/util/printchplenv --runtime --path`/main.o `$(CHPL_HOME)/util/config/compileline --libraries` $(DPCPP_LIBS) -o vc.dpcpp
	rm -rf tmp

opencl: vc.chpl build/libvc.opencl.a
	chpl $(CHPL_FLAGS) vc.h vc.chpl $(OPENCL_LIBS) -o vc.opencl

.PHONY: clean
clean:
	rm -rf build vc.cuda vc.hip vc.opencl
