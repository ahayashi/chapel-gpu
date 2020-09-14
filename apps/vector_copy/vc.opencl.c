#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define VERBOSE

#define MAX_SOURCE_SIZE (0x100000)

const char *getErrorString(cl_int error)
{
    switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}

#ifdef __cplusplus
extern "C" {
#endif
    void vcCUDA(float* A, float *B, int start, int end, int GPUN) {
	if (GPUN > 0) {
	    assert(end - start + 1 == GPUN);
#ifdef VERBOSE
	    printf("In vcOCL\n");
	    printf("\t GPUN: %d\n", GPUN);
	    printf("\t range: %d..%d\n", start, end);
#endif

        FILE *fp;
        char *source_str;
        size_t source_size;
        char str[1024];

        fp = fopen("vc.cl", "r");
        if (!fp) {
            fprintf(stderr, "Failed to load kernel.\n");
            exit(1);
        }
        source_str = (char*)malloc(MAX_SOURCE_SIZE);
        source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
        fclose( fp );

	    //printf("source: %s\n", source_str);

        // Get platform and device information
        cl_platform_id platform_id = NULL;
        cl_device_id device_ids[2];
        cl_uint ret_num_devices;
        cl_uint ret_num_platforms;
        cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
        if (ret != CL_SUCCESS) {
            printf("clGetPlatformIDs %s\n", getErrorString(ret));
        }

        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 2, device_ids, &ret_num_devices);
        //printf("device ID: %d, # of devices: %d\n", ret, ret_num_devices);
        int did = 0;
        char *env = getenv("OCL_DEVICE_NO");
        if (env) {
            did = atoi(env);
        }

        cl_device_id device_id = device_ids[did];
        size_t sret;
        clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(str), str, &sret);
		printf("clGetDeviceInfo = %ld, GPU %s\n", sret, str);

        // Create an OpenCL context
        cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
        if (ret != CL_SUCCESS) {
            printf("%s\n", getErrorString(ret));
        }

        // Create a command queue
        cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
        if (ret != CL_SUCCESS) {
            printf("%s\n", getErrorString(ret));
        }

        // Create memory buffers on the device for each vector
        cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, GPUN * sizeof(float), NULL, &ret);
        if (ret != CL_SUCCESS) {
            printf("%s\n", getErrorString(ret));
        }

        cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, GPUN * sizeof(float), NULL, &ret);
        if (ret != CL_SUCCESS) {
            printf("%s\n", getErrorString(ret));
        }

        cl_event h2d_event;
        ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, GPUN * sizeof(float), B + start, 0, NULL, &h2d_event);
        if (ret != CL_SUCCESS) {
            printf("%s\n", getErrorString(ret));
        } else {
            clWaitForEvents(1, &h2d_event);
        }

        // Create a program from the kernel source
        cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
        if (ret != CL_SUCCESS) {
            printf("%s\n", getErrorString(ret));
        }

        // Build the program
        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        if (ret != CL_SUCCESS) {
            printf("%s\n", getErrorString(ret));
        }

        // Create the OpenCL kernel
        cl_kernel kernel = clCreateKernel(program, "vc", &ret);
        if (ret != CL_SUCCESS) {
            printf("%s\n", getErrorString(ret));
        }

        // Set the arguments of the kernel
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
        if (ret != CL_SUCCESS) {
            printf("%s\n", getErrorString(ret));
        }

        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
        if (ret != CL_SUCCESS) {
            printf("%s\n", getErrorString(ret));
        }

        ret = clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&GPUN);
        if (ret != CL_SUCCESS) {
            printf("%s\n", getErrorString(ret));
        }

        // Execute the OpenCL kernel on the list
        size_t local_item_size = 64; // Divide work items into groups of 64
        size_t global_item_size = local_item_size * ((GPUN + local_item_size -1) / local_item_size); // Process the entire lists
        cl_event k_event;
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &k_event);
        if (ret != CL_SUCCESS) {
            printf("%s\n", getErrorString(ret));
        } else {
            clWaitForEvents(1, &k_event);
        }

        cl_event d2h_event;
        ret = clEnqueueReadBuffer(command_queue, a_mem_obj, CL_TRUE, 0, GPUN * sizeof(float), A + start, 0, NULL, &d2h_event);
        if (ret != CL_SUCCESS) {
            printf("%s\n", getErrorString(ret));
        }
        ret = clFinish(command_queue);
        if (ret != CL_SUCCESS) {
            printf("%s\n", getErrorString(ret));
        }
        cl_ulong time_start;
        cl_ulong time_end;

        // H2D
        clGetEventProfilingInfo(h2d_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(h2d_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
        printf("H2D time: %lf seconds \n", (time_end-time_start) / 1000000000.0);

        // Kernel
        clGetEventProfilingInfo(k_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(k_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
        printf("Kernel time: %lf seconds \n", (time_end-time_start) / 1000000000.0);
        // D2H
        clGetEventProfilingInfo(d2h_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(d2h_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
        printf("D2H time: %lf seconds \n", (time_end-time_start) / 1000000000.0);
    }
    }
#ifdef __cplusplus
}
#endif
