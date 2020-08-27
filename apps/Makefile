CHPL_GPU_MODULES=-M $(CHPL_GPU_HOME)/modules $(CHPL_GPU_HOME)/include/GPUAPI.h
CHPL_FLAGS=--fast $(CHPL_GPU_MODULES)

# CUDA
ifeq ($(USE_CUBLAS), yes)
  CUBLAS_LIB=-lcublas
endif
CUDA_LIBS=-L$(CHPL_GPU_HOME)/lib -lGPUAPICUDA_static -L$(CUDA_ROOT_DIR)/lib -lcudart $(CUBLAS_LIB)
# HIP
HIP_LIBS=-L$(CHPL_GPU_HOME)/lib -lGPUAPIHIP_static -L$(HIP_ROOT_DIR)/lib -lhip_hcc

# OpenCL
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	OPENCL_LIBS=-L$(CHPL_GPU_HOME)/lib -lGPUAPIOPENCL_static --ldflags '-framework OpenCL'
else
	OPENCL_LIBS=-L$(CHPL_GPU_HOME)/lib -lGPUAPIOPENCL_static -L$(subst libOpenCL.so,,$(OpenCL_LIBRARIES)) -lOpenCL
endif

# CPU

baseline: $(TARGET).baseline.chpl
	chpl --fast $< -o $(TARGET).cpu.$@

# CUDA

cuda: cuda.gpu cuda.hybrid cuda.hybrid.dist cuda.hybrid.dist.lowmid cuda.hybrid.dist.mid

build/lib$(TARGET).cuda.a: $(TARGET).cu
	rm -rf build &&	mkdir build && cd build && cmake .. && make

build/lib$(TARGET).kernel.cuda.a: $(TARGET).kernel.cu
	rm -rf build &&	mkdir build && cd build && cmake .. && make

cuda.%.mid: $(TARGET).%.mid.chpl $(TARGET).kernel.h build/lib$(TARGET).kernel.cuda.a
	chpl $(CHPL_FLAGS) $(TARGET).kernel.h $(TARGET).$*.mid.chpl -L./build -l$(TARGET).kernel.cuda $(CUDA_LIBS) -o $(TARGET).$@

cuda.%.lowmid: $(TARGET).%.lowmid.chpl $(TARGET).kernel.h build/lib$(TARGET).kernel.cuda.a
	chpl $(CHPL_FLAGS) $(TARGET).kernel.h $(TARGET).$*.lowmid.chpl -L./build -l$(TARGET).kernel.cuda $(CUDA_LIBS) -o $(TARGET).$@

cuda.%: $(TARGET).%.chpl $(TARGET).h build/lib$(TARGET).cuda.a
	chpl $(CHPL_FLAGS) $(TARGET).h $(TARGET).$*.chpl -L./build -l$(TARGET).cuda $(CUDA_LIBS) -o $(TARGET).$@

# HIP

hip: hip.gpu hip.hybrid hip.hybrid.dist hip.hybrid.dist.lowmid hip.hybrid.dist.mid

build/lib$(TARGET).hip.a: $(TARGET).cu
	rm -rf build &&	mkdir build && cd build && cmake .. && make

build/lib$(TARGET).kernel.hip.a: $(TARGET).kernel.cu
	rm -rf build &&	mkdir build && cd build && cmake .. && make

hip.%.mid: $(TARGET).%.mid.chpl $(TARGET).kernel.h  build/lib$(TARGET).kernel.hip.a
	chpl $(CHPL_FLAGS) $(TARGET).kernel.h $(TARGET).$*.mid.chpl -L./build -l$(TARGET).kernel.hip $(HIP_LIBS) -o $(TARGET).$@

hip.%.lowmid: $(TARGET).%.lowmid.chpl $(TARGET).kernel.h build/lib$(TARGET).kernel.hip.a
	chpl $(CHPL_FLAGS) $(TARGET).kernel.h $(TARGET).$*.lowmid.chpl -L./build -l$(TARGET).kernel.hip $(HIP_LIBS) -o $(TARGET).$@

hip.%: $(TARGET).%.chpl $(TARGET).h build/lib$(TARGET).hip.a
	chpl $(CHPL_FLAGS) $(TARGET).h $(TARGET).$*.chpl -L./build -l$(TARGET).hip $(HIP_LIBS) -o $(TARGET).$@

# OpenCL (MID and LOW-MID are not supported so far)

opencl: opencl.gpu opencl.hybrid opencl.hybrid.dist

build/lib$(TARGET).opencl.a: $(TARGET).opencl.c
	rm -rf build &&	mkdir build && cd build && cmake .. && make

opencl.%: $(TARGET).%.chpl $(TARGET).h build/lib$(TARGET).opencl.a
	chpl $(CHPL_FLAGS) $(TARGET).h $(TARGET).$*.chpl -L./build -l$(TARGET).opencl $(OPENCL_LIBS) -o $(TARGET).$@

.PHONY: clean
clean:
	rm -rf ./build $(TARGET).cpu.baseline $(TARGET).cuda.* $(TARGET).hip.* $(TARGET).opencl.gpu $(TARGET).opencl.hybrid*
