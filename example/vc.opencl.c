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

#undef PROF

#define MAX_SOURCE_SIZE (0x100000)

#ifdef __cplusplus
extern "C" {
#endif

    extern char* openclGetErrorString(cl_int);

    void vcGPU(float* A, float *B, int start, int end, int GPUN) {
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

        // Get platform and device information
        cl_platform_id platform_id = NULL;
        cl_device_id device_ids[2];
        cl_uint ret_num_devices;
        cl_uint ret_num_platforms;
        cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
        if (ret != CL_SUCCESS) {
            printf("clGetPlatformIDs %s\n", openclGetErrorString(ret));
        }

        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 2, device_ids, &ret_num_devices);
        int did = 0;
        char *env = getenv("OCL_DEVICE_NO");
        if (env) {
            did = atoi(env);
        }

        cl_device_id device_id = device_ids[did];
        size_t sret;
        clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(str), str, &sret);
#ifdef PROF
		printf("clGetDeviceInfo = %ld, GPU %s\n", sret, str);
#endif
        // Create an OpenCL context
        cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
        if (ret != CL_SUCCESS) {
            printf("%s\n", openclGetErrorString(ret));
        }

        // Create a command queue
        cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
        if (ret != CL_SUCCESS) {
            printf("%s\n", openclGetErrorString(ret));
        }

        // Create memory buffers on the device for each vector
        cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, GPUN * sizeof(float), NULL, &ret);
        if (ret != CL_SUCCESS) {
            printf("%s\n", openclGetErrorString(ret));
        }

        cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, GPUN * sizeof(float), NULL, &ret);
        if (ret != CL_SUCCESS) {
            printf("%s\n", openclGetErrorString(ret));
        }

        cl_event h2d_event;
        ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, GPUN * sizeof(float), B + start, 0, NULL, &h2d_event);
        if (ret != CL_SUCCESS) {
            printf("%s\n", openclGetErrorString(ret));
        } else {
            clWaitForEvents(1, &h2d_event);
        }

        // Create a program from the kernel source
        cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
        if (ret != CL_SUCCESS) {
            printf("%s\n", openclGetErrorString(ret));
        }

        // Build the program
        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        if (ret != CL_SUCCESS) {
            printf("%s\n", openclGetErrorString(ret));
        }

        // Create the OpenCL kernel
        cl_kernel kernel = clCreateKernel(program, "vc", &ret);
        if (ret != CL_SUCCESS) {
            printf("%s\n", openclGetErrorString(ret));
        }

        // Set the arguments of the kernel
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
        if (ret != CL_SUCCESS) {
            printf("%s\n", openclGetErrorString(ret));
        }

        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
        if (ret != CL_SUCCESS) {
            printf("%s\n", openclGetErrorString(ret));
        }

        ret = clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&GPUN);
        if (ret != CL_SUCCESS) {
            printf("%s\n", openclGetErrorString(ret));
        }

        // Execute the OpenCL kernel on the list
        size_t local_item_size = 64; // Divide work items into groups of 64
        size_t global_item_size = local_item_size * ((GPUN + local_item_size -1) / local_item_size); // Process the entire lists
        cl_event k_event;
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &k_event);
        if (ret != CL_SUCCESS) {
            printf("%s\n", openclGetErrorString(ret));
        } else {
            clWaitForEvents(1, &k_event);
        }

        cl_event d2h_event;
        ret = clEnqueueReadBuffer(command_queue, a_mem_obj, CL_TRUE, 0, GPUN * sizeof(float), A + start, 0, NULL, &d2h_event);
        if (ret != CL_SUCCESS) {
            printf("%s\n", openclGetErrorString(ret));
        }
        ret = clFinish(command_queue);
        if (ret != CL_SUCCESS) {
            printf("%s\n", openclGetErrorString(ret));
        }
#if PROF
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
#endif
    }
#ifdef __cplusplus
}
#endif
