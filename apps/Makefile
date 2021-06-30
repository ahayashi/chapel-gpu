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
# DPC++
DPCPP_LIBS=$(CHPL_GPU_LIB_PATH) -lGPUAPIDPCPP_static

# OpenCL
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	OPENCL_LIBS=$(CHPL_GPU_LIB_PATH) -lGPUAPIOPENCL_static --ldflags '-framework OpenCL'
else
	OPENCL_LIBS=$(CHPL_GPU_LIB_PATH) -lGPUAPIOPENCL_static -L$(subst libOpenCL.so,,$(OpenCL_LIBRARIES)) -lOpenCL
endif

# CPU

baseline: $(TARGET).baseline.chpl
	chpl --fast $< -o $(TARGET).cpu.$@

# CUDA

cuda: cuda.gpu cuda.hybrid cuda.hybrid.dist cuda.hybrid.dist.midlow cuda.hybrid.dist.mid

build/lib$(TARGET).cuda.a: $(TARGET).cu
	rm -rf build &&	mkdir build && cd build && cmake .. && make

build/lib$(TARGET).kernel.cuda.a: $(TARGET).kernel.cu
	rm -rf build &&	mkdir build && cd build && cmake .. && make

cuda.%.mid: $(TARGET).%.mid.chpl $(TARGET).kernel.h build/lib$(TARGET).kernel.cuda.a
	chpl $(CHPL_FLAGS) $(TARGET).kernel.h $(TARGET).$*.mid.chpl -L./build -l$(TARGET).kernel.cuda $(CUDA_LIBS) -o $(TARGET).$@

cuda.%.midlow: $(TARGET).%.midlow.chpl $(TARGET).kernel.h build/lib$(TARGET).kernel.cuda.a
	chpl $(CHPL_FLAGS) $(TARGET).kernel.h $(TARGET).$*.midlow.chpl -L./build -l$(TARGET).kernel.cuda $(CUDA_LIBS) -o $(TARGET).$@

cuda.%: $(TARGET).%.chpl $(TARGET).h build/lib$(TARGET).cuda.a
	chpl $(CHPL_FLAGS) $(TARGET).h $(TARGET).$*.chpl -L./build -l$(TARGET).cuda $(CUDA_LIBS) -o $(TARGET).$@

# HIP

hip: hip.gpu hip.hybrid hip.hybrid.dist hip.hybrid.dist.midlow hip.hybrid.dist.mid

build/lib$(TARGET).hip.a: $(TARGET).cu
	rm -rf build &&	mkdir build && cd build && cmake .. && make

build/lib$(TARGET).kernel.hip.a: $(TARGET).kernel.cu
	rm -rf build &&	mkdir build && cd build && cmake .. && make

hip.%.mid: $(TARGET).%.mid.chpl $(TARGET).kernel.h  build/lib$(TARGET).kernel.hip.a
	chpl $(CHPL_FLAGS) $(TARGET).kernel.h $(TARGET).$*.mid.chpl -L./build -l$(TARGET).kernel.hip $(HIP_LIBS) -o $(TARGET).$@

hip.%.midlow: $(TARGET).%.midlow.chpl $(TARGET).kernel.h build/lib$(TARGET).kernel.hip.a
	chpl $(CHPL_FLAGS) $(TARGET).kernel.h $(TARGET).$*.midlow.chpl -L./build -l$(TARGET).kernel.hip $(HIP_LIBS) -o $(TARGET).$@

hip.%: $(TARGET).%.chpl $(TARGET).h build/lib$(TARGET).hip.a
	chpl $(CHPL_FLAGS) $(TARGET).h $(TARGET).$*.chpl -L./build -l$(TARGET).hip $(HIP_LIBS) -o $(TARGET).$@

# OpenCL (MID and LOW-MID are not supported so far)

opencl: opencl.gpu opencl.hybrid opencl.hybrid.dist

build/lib$(TARGET).opencl.a: $(TARGET).opencl.c
	rm -rf build &&	mkdir build && cd build && cmake .. && make

opencl.%: $(TARGET).%.chpl $(TARGET).h build/lib$(TARGET).opencl.a
	chpl $(CHPL_FLAGS) $(TARGET).h $(TARGET).$*.chpl -L./build -l$(TARGET).opencl $(OPENCL_LIBS) -o $(TARGET).$@

#DPC++

dpcpp: dpcpp.gpu dpcpp.hybrid dpcpp.hybrid.dist dpcpp.hybrid.dist.midlow dpcpp.hybrid.dist.mid

build/lib$(TARGET).dpcpp.a: $(TARGET).cu
	rm -rf build &&	mkdir build && cd build && cmake .. && make

build/lib$(TARGET).kernel.dpcpp.a: $(TARGET).kernel.cu
	rm -rf build &&	mkdir build && cd build && cmake .. && make

dpcpp.%.mid: $(TARGET).%.mid.chpl $(TARGET).kernel.h  build/lib$(TARGET).kernel.dpcpp.a
	rm -rf tmp
	-chpl --savec tmp $(CHPL_FLAGS) $(TARGET).$*.mid.chpl
	dpcpp build/$(TARGET).kernel.dp.cpp tmp/$(TARGET).$*.mid.tmp.o $(CHPL_HOME)/lib/`$(CHPL_HOME)/util/printchplenv --runtime --path`/main.o `$(CHPL_HOME)/util/config/compileline --libraries` $(DPCPP_LIBS) -o $(TARGET).$@
	rm -rf tmp

dpcpp.%.midlow: $(TARGET).%.midlow.chpl $(TARGET).kernel.h  build/lib$(TARGET).kernel.dpcpp.a
	rm -rf tmp
	-chpl --savec tmp $(CHPL_FLAGS) $(TARGET).$*.midlow.chpl
	dpcpp build/$(TARGET).kernel.dp.cpp tmp/$(TARGET).$*.midlow.tmp.o $(CHPL_HOME)/lib/`$(CHPL_HOME)/util/printchplenv --runtime --path`/main.o `$(CHPL_HOME)/util/config/compileline --libraries` $(DPCPP_LIBS) -o $(TARGET).$@
	rm -rf tmp

dpcpp.%: $(TARGET).%.chpl $(TARGET).h build/lib$(TARGET).dpcpp.a
	rm -rf tmp
	-chpl --savec tmp $(CHPL_FLAGS) $(TARGET).$*.chpl
	dpcpp build/$(TARGET).dp.cpp tmp/$(TARGET).$*.tmp.o $(CHPL_HOME)/lib/`$(CHPL_HOME)/util/printchplenv --runtime --path`/main.o `$(CHPL_HOME)/util/config/compileline --libraries` $(DPCPP_LIBS) -o $(TARGET).$@
	rm -rf tmp

.PHONY: clean
clean:
	rm -rf ./build $(TARGET).cpu.baseline $(TARGET).cuda.* $(TARGET).hip.* $(TARGET).opencl.gpu $(TARGET).opencl.hybrid* $(TARGET).dpcpp.*
