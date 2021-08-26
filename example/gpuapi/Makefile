CHPL_GPU_MODULES=-M $(CHPL_GPU_HOME)/modules $(CHPL_GPU_HOME)/include/GPUAPI.h
CHPL_FLAGS=--fast $(CHPL_GPU_MODULES)
CHPL_GPU_LIB_PATH=-L$(CHPL_GPU_HOME)/lib -L$(CHPL_GPU_HOME)/lib64
CHPL_GPU_CMAKE_PATH=$(CHPL_GPU_HOME)/share/cmake/ChapelGPUAPI

# CUDA
ifeq ($(USE_CUBLAS), yes)
  CUBLAS_LIB=-lcublas
endif
CUDA_LIBS=$(CHPL_GPU_LIB_PATH) -lGPUAPICUDA_static -L$(CUDA_ROOT_DIR)/lib -lcudart $(CUBLAS_LIB)
# HIP
HIP_LIBS=$(CHPL_GPU_LIB_PATH) -lGPUAPIHIP_static -L$(HIP_ROOT_DIR)/lib -lhip_hcc
# DPC++
DPCPP_LIBS=$(CHPL_GPU_LIB_PATH) -lGPUAPIDPCPP_static $(DBLAS_LIB)
DPCPP_FLAGS=-DTHREADS_PER_BLOCK=256 $(DBLAS_FLAG)

# CUDA

build/lib$(TARGET).kernel.cuda.a: $(TARGET).kernel.cu
	rm -rf build &&	mkdir build && cd build && cmake .. -DCMAKE_MODULE_PATH=$(CHPL_GPU_CMAKE_PATH) && make

cuda: $(TARGET).chpl $(TARGET).kernel.h build/lib$(TARGET).kernel.cuda.a
	chpl $(CHPL_FLAGS) $(TARGET).kernel.h $(TARGET).chpl -L./build -l$(TARGET).kernel.cuda $(CUDA_LIBS) -o $(TARGET).$@

# HIP

build/lib$(TARGET).kernel.hip.a: $(TARGET).kernel.cu
	rm -rf build &&	mkdir build && cd build && cmake .. -DCMAKE_MODULE_PATH=$(CHPL_GPU_CMAKE_PATH) && make

hip: $(TARGET).chpl $(TARGET).kernel.h  build/lib$(TARGET).kernel.hip.a
	chpl $(CHPL_FLAGS) $(TARGET).kernel.h $(TARGET).chpl -L./build -l$(TARGET).kernel.hip $(HIP_LIBS) -o $(TARGET).$@

# DPC++

build/lib$(TARGET).kernel.dpcpp.a: $(TARGET).kernel.cu
	rm -rf build &&	mkdir build && cd build && cmake .. -DCMAKE_MODULE_PATH=$(CHPL_GPU_CMAKE_PATH) && make

dpcpp: $(TARGET).chpl $(TARGET).kernel.h build/lib$(TARGET).kernel.dpcpp.a
	rm -rf tmp
	-chpl --savec tmp $(CHPL_FLAGS) $(TARGET).chpl
	dpcpp $(DPCPP_FLAGS) build/$(TARGET).kernel.dp.cpp tmp/$(TARGET).tmp.o $(CHPL_HOME)/lib/`$(CHPL_HOME)/util/printchplenv --runtime --path`/main.o `$(CHPL_HOME)/util/config/compileline --libraries` $(DPCPP_LIBS) -o $(TARGET).dpcpp
	rm -rf tmp

.PHONY: clean
clean:
	rm -rf ./build $(TARGET).cuda $(TARGET).hip $(TARGET).dpcpp
