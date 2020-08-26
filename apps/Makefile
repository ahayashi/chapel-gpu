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
	OPENCL_LIBS=-L$(CHPL_GPU_HOME)/lib -lGPUAPIOPENCL_static -L./build -l$(TARGET).opencl --ldflags '-framework OpenCL'
else
	OPENCL_LIBS=-L$(CHPL_GPU_HOME)/lib -lGPUAPIOPENCL_static -L./build -l$(TARGET).opencl -L$(subst libOpenCL.so,,$(OpenCL_LIBRARIES)) -lOpenCL
endif

all: cudahybrid.dist

gpu: $(TARGET).cu
	rm -rf build &&	mkdir build && cd build && cmake .. && make

cuda%: gpu $(TARGET).%.chpl
ifneq (,$(filter $*, hybrid.dist.mid hybrid.dist.lowmid))
		chpl $(CHPL_FLAGS) $(TARGET).h $(TARGET).$*.chpl -L./build -l$(TARGET).cuda $(CUDA_LIBS) -o $(TARGET).cuda.$*
else
		chpl $(CHPL_FLAGS) $(TARGET).kernel.h $(TARGET).$*.chpl -L./build -l$(TARGET).kernel.cuda $(CUDA_LIBS) -o $(TARGET).cuda.$*
endif

hip%: gpu $(TARGET).%.chpl
ifneq (,$(filter $*, hybrid.dist.mid hybrid.dist.lowmid))
	chpl $(CHPL_FLAGS) $(TARGET).h $(TARGET).$*.chpl -L./build -l$(TARGET).hip $(HIP_LIBS) -o $(TARGET).hip.$*
else
	chpl $(CHPL_FLAGS) $(TARGET).kernel.h $(TARGET).$*.chpl -L./build -l$(TARGET).kernel.hip $(HIP_LIBS) -o $(TARGET).hip.$*
endif

opencl%: gpu $(TARGET).%.chpl
	chpl $(CHPL_FLAGS) $(TARGET).h $(TARGET).$*.chpl $(OPNCL_LIBS) -o $(TARGET).opencl.$*
