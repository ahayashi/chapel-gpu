CHPL_GPU_MODULES=-M $(CHPL_GPU_HOME)/modules $(CHPL_GPU_HOME)/include/GPUAPI.h
CHPL_FLAGS=--fast $(CHPL_GPU_MODULES)

# CUDA
ifeq ($(USE_CUBLAS), yes)
  CUBLAS_LIB=-lcublas
endif
CUDA_LIBS=-L$(CHPL_GPU_HOME)/lib -lGPUAPICUDA_static -L./build -l$(TARGET).cuda -L$(CUDA_ROOT_DIR)/lib -lcudart $(CUBLAS_LIB)
# HIP
HIP_LIBS=-L$(CHPL_GPU_HOME)/lib -lGPUAPIHIP_static -L./build -l$(TARGET).hip -L$(HIP_ROOT_DIR)/lib -lhip_hcc

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
	chpl $(CHPL_FLAGS) $(TARGET).h $(TARGET).$*.chpl $(CUDA_LIBS) -o $(TARGET).cuda.$*

hip%: gpu $(TARGET).%.chpl
	chpl $(CHPL_FLAGS) $(TARGET).h $(TARGET).$*.chpl $(HIP_LIBS) -o $(TARGET).hip.$*

opencl%: gpu $(TARGET).%.chpl
	chpl $(CHPL_FLAGS) $(TARGET).h $(TARGET).$*.chpl $(OPNCL_LIBS) -o $(TARGET).opencl.$*
