CHPL_GPU_MODULES=-M $(CHPL_GPU_HOME)/modules $(CHPL_GPU_HOME)/include/GPUAPI.h
CHPL_FLAGS=--fast $(CHPL_GPU_MODULES)
CHPL_GPU_LIB_PATH=-L$(CHPL_GPU_HOME)/lib -L$(CHPL_GPU_HOME)/lib64

# CUDA
ifeq ($(USE_CUBLAS), yes)
  CUBLAS_LIB=-lcublas
endif
CUDA_LIBS=$(CHPL_GPU_LIB_PATH) -lGPUAPICUDA_static -L$(CUDA_ROOT_DIR)/lib -lcudart $(CUBLAS_LIB)
# HIP
HIP_LIBS=$(CHPL_GPU_LIB_PATH) -lGPUAPIHIP_static -L$(HIP_ROOT_DIR)/lib -lhip_hcc

# CUDA

build/lib$(TARGET).kernel.cuda.a: $(TARGET).kernel.cu
	rm -rf build &&	mkdir build && cd build && cmake .. && make

cuda: $(TARGET).chpl $(TARGET).kernel.h build/lib$(TARGET).kernel.cuda.a
	chpl $(CHPL_FLAGS) $(TARGET).kernel.h $(TARGET).chpl -L./build -l$(TARGET).kernel.cuda $(CUDA_LIBS) -o $(TARGET).$@

# HIP

build/lib$(TARGET).kernel.hip.a: $(TARGET).kernel.cu
	rm -rf build &&	mkdir build && cd build && cmake .. && make

hip: $(TARGET).%.mid.chpl $(TARGET).kernel.h  build/lib$(TARGET).kernel.hip.a
	chpl $(CHPL_FLAGS) $(TARGET).kernel.h $(TARGET).$*.mid.chpl -L./build -l$(TARGET).kernel.hip $(HIP_LIBS) -o $(TARGET).$@

.PHONY: clean
clean:
	rm -rf ./build $(TARGET).cuda $(TARGET).hip
