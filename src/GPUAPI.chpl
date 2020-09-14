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

    config param debugGPUAPI = false;

    extern proc GetDeviceCount(ref count: int(32));
    extern proc GetDevice(ref device: int(32));
    extern proc SetDevice(device: int(32));

    extern proc ProfilerStart();
    extern proc ProfilerStop();

    extern proc DeviceSynchronize();

    extern proc Malloc(ref devPtr: c_void_ptr, size: size_t);
    extern proc Memcpy(dst: c_void_ptr, src: c_void_ptr, count: size_t, kind: int);
    extern proc Free(devPtr: c_void_ptr);

    class GPUArray {
      var h2d: bool;
      var d2h: bool;
      var devPtr: c_void_ptr;
      var hosPtr: c_void_ptr;
      var size: size_t;
      var sizeInBytes: size_t;

      proc init(ref arr) {
        // Properties
        this.h2d = true;
        this.d2h = true;
        // Low-level info
        this.devPtr = nil;
        this.hosPtr = c_ptrTo(arr);
        // size info
        size = arr.size: size_t;
        sizeInBytes = (((arr.size: size_t) * c_sizeof(arr.eltType)) : size_t);
        this.complete();
        // allocation
        Malloc(devPtr, sizeInBytes);
        if (debugGPUAPI) { writeln("malloc'ed: ", devPtr, " sizeInBytes: ", sizeInBytes); }
      }

      proc toDevice() {
        if (this.h2d) {
          Memcpy(this.dPtr(), this.hPtr(), this.sizeInBytes, 0);
          if (debugGPUAPI) { writeln("h2d : ", this.hPtr(), " -> ", this.dPtr(), " transBytes: ", this.sizeInBytes); }
        } else {
          if (debugGPUAPI) { writeln("h2d ignored"); }
        }
      }

      proc fromDevice() {
        if (this.d2h) {
          Memcpy(this.hPtr(), this.dPtr(), this.sizeInBytes, 1);
          if (debugGPUAPI) { writeln("d2h : ", this.dPtr(), " -> ", this.hPtr(), " transBytes: ", this.sizeInBytes); }
        } else {
          if (debugGPUAPI) { writeln("d2h ignored"); }
        }
      }

      proc free() {
        Free(this.dPtr());
        if (debugGPUAPI) { writeln("free : ", this.dPtr()); }
      }

      proc dPtr(): c_void_ptr {
        return devPtr;
      }

      proc hPtr(): c_void_ptr {
        return hosPtr;
      }
    }

    proc toDevice(args: GPUArray ...?n) {
      for ga in args {
        ga.toDevice();
      }
    }

    proc fromDevice(args: GPUArray ...?n) {
      for ga in args {
        ga.fromDevice();
      }
    }

    proc free(args: GPUArray ...?n) {
      for ga in args {
        ga.free();
      }
    }
}