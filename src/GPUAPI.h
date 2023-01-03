#ifndef _GPU_API_H
#define _GPU_API_H
#ifdef __cplusplus
extern "C" {
#endif
void GetDeviceCount(int*);
void GetDevice(int*);
void SetDevice(int);
void ProfilerStart(void);
void ProfilerStop(void);
void DeviceSynchronize(void);
void Malloc(void**, size_t);
void MallocPtr(void***, size_t);
void MallocPtrPtr(void****, size_t);
void MallocPitch(void**, size_t*, size_t, size_t);
void Memcpy(void*, void*, size_t, int);
void Memcpy2D(void*, size_t, void*, size_t, size_t, size_t, int);
void Free(void*);
#ifdef __cplusplus
}
#endif
#endif
