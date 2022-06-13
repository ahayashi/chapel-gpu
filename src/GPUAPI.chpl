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
    use CTypes;

    config param debugGPUAPI = false;

    extern proc GetDeviceCount(ref count: int(32));
    extern proc GetDevice(ref device: int(32));
    extern proc SetDevice(device: int(32));

    extern proc ProfilerStart();
    extern proc ProfilerStop();

    extern proc DeviceSynchronize();

    // cudaMalloc
    extern proc Malloc(ref devPtr: c_void_ptr, size: c_size_t);
    extern proc MallocPtr(ref devPtr: c_ptr(c_void_ptr), size: c_size_t);
    extern proc MallocPtrPtr(ref devPtr: c_ptr(c_ptr(c_void_ptr)), size: c_size_t);
    inline proc Malloc(ref devPtr: c_ptr(c_void_ptr), size: c_size_t) { MallocPtr(devPtr, size); };
    inline proc Malloc(ref devPtr: c_ptr(c_ptr(c_void_ptr)), size: c_size_t) { MallocPtrPtr(devPtr, size); };

    // cudaMallocPitch
    extern proc MallocPitch(ref devPtr: c_void_ptr, ref pitch: c_size_t, width: c_size_t, height: c_size_t);

    extern proc Memcpy(dst: c_void_ptr, src: c_void_ptr, count: c_size_t, kind: int);
    extern proc Memcpy2D(dst: c_void_ptr, dpitch: c_size_t, src: c_void_ptr, spitch: c_size_t, width: c_size_t, height: c_size_t, kind: int);
    extern proc Free(devPtr: c_void_ptr);

    class GPUArray {
      var devPtr: c_void_ptr;
      var hosPtr: c_void_ptr;
      var size: c_size_t;
      var sizeInBytes: c_size_t;
      var pitched: bool;
      var hpitch: c_size_t;
      var dpitch: c_size_t;
      var height: c_size_t;

      proc init(ref arr, pitched=false) {
        // Low-level info
        this.devPtr = nil;
        this.hosPtr = c_ptrTo(arr);
        // size info
        size = arr.size: c_size_t;
        sizeInBytes = (((arr.size: c_size_t) * c_sizeof(arr.eltType)) : c_size_t);
        this.pitched = pitched;
        this.complete();
        if (!pitched) {
            // allocation
            Malloc(devPtr, sizeInBytes);
            if (arr.rank == 2) {
               this.hpitch = arr.domain.dim(1).size:c_size_t * c_sizeof(arr.eltType);
               this.dpitch = this.hpitch;
            }
        } else if (arr.rank == 2) {
            this.hpitch = arr.domain.dim(1).size:c_size_t * c_sizeof(arr.eltType);
            // allocation
            MallocPitch(devPtr, this.dpitch, this.hpitch, arr.domain.dim(0).size:c_size_t);
            this.height = arr.domain.dim(0).size:c_size_t;
        } else {
            writeln("GPU Array allocation error: pitched=true only works with 2D array, but the given rank is ", arr.rank);
            exit();
        }
        if (debugGPUAPI) { writeln("malloc'ed: ", devPtr, " sizeInBytes: ", sizeInBytes); }
      }

      proc deinit() {
          Free(this.dPtr());
      }

      inline proc toDevice() {
          if (this.pitched == false) {
              Memcpy(this.dPtr(), this.hPtr(), this.sizeInBytes, 0);
          } else {
              Memcpy2D(this.dPtr(), this.dpitch, this.hPtr(), this.hpitch, this.hpitch, this.height, 0);
          }
          if (debugGPUAPI) { writeln("h2d : ", this.hPtr(), " -> ", this.dPtr(), " transBytes: ", this.sizeInBytes); }
      }

      inline proc fromDevice() {
          if (this.pitched == false) {
              Memcpy(this.hPtr(), this.dPtr(), this.sizeInBytes, 1);
          } else {
              Memcpy2D(this.hPtr(), this.hpitch, this.dPtr(), this.dpitch, this.hpitch, this.height, 1);
          }
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
      var elemSizes: [0..#nRows] c_size_t;
      var size: c_size_t;
      var sizeInBytes: c_size_t;

      // args: iterable
      // class C {
      //   var n: int;
      //   proc init(_n: int) { n = _n;  }
      //   var x: [0..#n] int;
      // }
      // var Cs = [new C(256), new C(512)];
      // var dCs = new GPUJaggedArray(Cs.x);
      proc init(args) {
        this.nRows = 0;
        for i in args {
            this.nRows = this.nRows + 1;
        }
        this.complete();
        var idx = 0;
        for i in args {
          const elemSize = i.size:c_size_t * c_sizeof(i.eltType);
          Malloc(this.devPtrs[idx], elemSize);
          this.hosPtrs[idx] = c_ptrTo(i);
          this.elemSizes[idx] = elemSize;
          idx = idx + 1;
        }
        Malloc(this.devPtr, nRows:c_size_t * c_sizeof(c_ptr(c_void_ptr)));
      }

      // var dCs = new GPUJaggedArray(Cs[0].x, Cs[1].x, ...);
      proc init(args ...?n) where n>=2 {
        this.nRows = n;
        this.complete();
        var idx: int = 0;
        for arg in args {
          const elemSize = arg.size: c_size_t * c_sizeof(arg.eltType);
          Malloc(this.devPtrs[idx], elemSize);
          this.hosPtrs[idx] = c_ptrTo(arg);
          this.elemSizes[idx] = elemSize;
          idx = idx + 1;
        }
        Malloc(this.devPtr, nRows:c_size_t * c_sizeof(c_ptr(c_void_ptr)));
      }

      proc deinit() {
        for i in 0..#nRows {
          Free(this.devPtrs[i]);
        }
        Free(this.devPtr);
      }

      inline proc toDevice() {
        for i in 0..#nRows {
          Memcpy(this.devPtrs[i], this.hosPtrs[i], elemSizes[i], 0);
        }
        Memcpy(this.devPtr, c_ptrTo(this.devPtrs), nRows:c_size_t * c_sizeof(c_ptr(c_void_ptr)), 0);
      }

      inline proc fromDevice() {
        for i in 0..#nRows {
          Memcpy(this.hosPtrs[i], this.devPtrs[i], elemSizes[i], 1);
        }
      }

      inline proc dPtr(): c_ptr(c_void_ptr) {
        return devPtr;
      }
    }
}
