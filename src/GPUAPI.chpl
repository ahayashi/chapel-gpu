
/*
 * Copyright (c) 2019, Rice University
 * Copyright (c) 2019, Georgia Institute of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

module GPUAPI {
    use SysCTypes;
    use CPtr;

    config param debugGPUAPI = false;

    extern proc GetDeviceCount(ref count: int(32));
    extern proc GetDevice(ref device: int(32));
    extern proc SetDevice(device: int(32));

    extern proc ProfilerStart();
    extern proc ProfilerStop();

    extern proc DeviceSynchronize();

    // cudaMalloc
    extern proc Malloc(ref devPtr: c_void_ptr, size: size_t);
    extern proc MallocPtr(ref devPtr: c_ptr(c_void_ptr), size: size_t);
    extern proc MallocPtrPtr(ref devPtr: c_ptr(c_ptr(c_void_ptr)), size: size_t);
    inline proc Malloc(ref devPtr: c_ptr(c_void_ptr), size: size_t) { MallocPtr(devPtr, size); };
    inline proc Malloc(ref devPtr: c_ptr(c_ptr(c_void_ptr)), size: size_t) { MallocPtrPtr(devPtr, size); };

    // cudaMallocPitch
    extern proc MallocPitch(ref devPtr: c_void_ptr, ref pitch: size_t, width: size_t, height: size_t);

    extern proc Memcpy(dst: c_void_ptr, src: c_void_ptr, count: size_t, kind: int);
    extern proc Memcpy2D(dst: c_void_ptr, dpitch: size_t, src: c_void_ptr, spitch: size_t, width: size_t, height: size_t, kind: int);
    extern proc Free(devPtr: c_void_ptr);

    class GPUArray {
      var devPtr: c_void_ptr;
      var hosPtr: c_void_ptr;
      var size: size_t;
      var sizeInBytes: size_t;
      var pitch: size_t;

      proc init(ref arr) {
        // Low-level info
        this.devPtr = nil;
        this.hosPtr = c_ptrTo(arr);
        // size info
        size = arr.size: size_t;
        sizeInBytes = (((arr.size: size_t) * c_sizeof(arr.eltType)) : size_t);
        if (arr.rank == 2) {
            pitch = arr.domain.dim(1).size:size_t * c_sizeof(arr.eltType);
        }
        this.complete();
        // allocation
        Malloc(devPtr, sizeInBytes);
        if (debugGPUAPI) { writeln("malloc'ed: ", devPtr, " sizeInBytes: ", sizeInBytes); }
      }

      inline proc toDevice() {
          Memcpy(this.dPtr(), this.hPtr(), this.sizeInBytes, 0);
          if (debugGPUAPI) { writeln("h2d : ", this.hPtr(), " -> ", this.dPtr(), " transBytes: ", this.sizeInBytes); }
      }

      inline proc fromDevice() {
          Memcpy(this.hPtr(), this.dPtr(), this.sizeInBytes, 1);
          if (debugGPUAPI) { writeln("d2h : ", this.dPtr(), " -> ", this.hPtr(), " transBytes: ", this.sizeInBytes); }
      }

      inline proc free() {
        Free(this.dPtr());
        if (debugGPUAPI) { writeln("free : ", this.dPtr()); }
      }

      inline proc dPtr(): c_void_ptr {
        return devPtr;
      }

      inline proc hPtr(): c_void_ptr {
        return hosPtr;
      }
    }

    inline proc toDevice(args: GPUArray ...?n) {
      for ga in args {
        ga.toDevice();
      }
    }

    inline proc fromDevice(args: GPUArray ...?n) {
      for ga in args {
        ga.fromDevice();
      }
    }

    inline proc free(args: GPUArray ...?n) {
      for ga in args {
        ga.free();
      }
    }

    class GPUJaggedArray {
      var devPtr: c_ptr(c_void_ptr);
      var nRows: int;
      var hosPtrs: [0..#nRows] c_void_ptr;
      var devPtrs: [0..#nRows] c_void_ptr;
      var elemSizes: [0..#nRows] size_t;
      var size: size_t;
      var sizeInBytes: size_t;

      proc init(args) {
        this.nRows = 0;
        for i in args {
            this.nRows = this.nRows + 1;
        }
        this.complete();
        var idx = 0;
        for i in args {
          const elemSize = i.size:size_t * c_sizeof(i.eltType);
          Malloc(this.devPtrs[idx], elemSize);
          this.hosPtrs[idx] = c_ptrTo(i);
          this.elemSizes[idx] = elemSize;
          idx = idx + 1;
        }
        Malloc(this.devPtr, nRows:size_t * c_sizeof(c_ptr(c_void_ptr)));
      }

      proc init(args ...?n) where n>=2 {
        this.nRows = n;
        this.complete();
        var idx: int = 0;
        for arg in args {
          var array = [i in arg] i;
          const elemSize = array.size: size_t * c_sizeof(array.eltType);
          Malloc(this.devPtrs[idx], elemSize);
          this.hosPtrs[idx] = c_ptrTo(array);
          this.elemSizes[idx] = elemSize;
          // temporary
          Memcpy(this.devPtrs[idx], this.hosPtrs[idx], elemSizes[idx], 0);
          idx = idx + 1;
        }
        Malloc(this.devPtr, nRows:size_t * c_sizeof(c_ptr(c_void_ptr)));
        Memcpy(this.devPtr, c_ptrTo(this.devPtrs), nRows:size_t * c_sizeof(c_ptr(c_void_ptr)), 0);
        //toDevice();
      }

      inline proc toDevice() {
        for i in 0..#nRows {
          Memcpy(this.devPtrs[i], this.hosPtrs[i], elemSizes[i], 0);
        }
        Memcpy(this.devPtr, c_ptrTo(this.devPtrs), nRows:size_t * c_sizeof(c_ptr(c_void_ptr)), 0);
      }

      inline proc dPtr(): c_ptr(c_void_ptr) {
        return devPtr;
      }
    }
}