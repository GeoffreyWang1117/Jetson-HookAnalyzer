#!/usr/bin/env python3
"""
Test YOLOv8 TensorRT inference
"""

import tensorrt as trt
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TensorRTInference:
    """Simple TensorRT inference wrapper"""

    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None

        self._load_engine()
        self._allocate_buffers()

    def _load_engine(self):
        """Load TensorRT engine from file"""
        print(f"Loading engine from {self.engine_path}")

        with open(self.engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Failed to load engine")

        self.context = self.engine.create_execution_context()
        print(f"✓ Engine loaded successfully")

    def _allocate_buffers(self):
        """Allocate GPU memory for inputs/outputs"""
        self.stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            shape = self.engine.get_tensor_shape(tensor_name)

            # Calculate size
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    'name': tensor_name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'dtype': dtype
                })
                print(f"Input  {i}: {tensor_name} {shape} {dtype}")
            else:
                self.outputs.append({
                    'name': tensor_name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'dtype': dtype
                })
                print(f"Output {i}: {tensor_name} {shape} {dtype}")

    def infer(self, input_data):
        """Run inference"""
        # Copy input to GPU
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )

        # Set input shape
        self.context.set_input_shape(self.inputs[0]['name'], input_data.shape)

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy output from GPU
        outputs = []
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
            outputs.append(output['host'].copy())

        self.stream.synchronize()

        return outputs

    def benchmark(self, input_shape=(1, 3, 640, 640), warmup=10, iterations=100):
        """Benchmark inference performance"""
        print(f"\n=== Benchmarking ===")
        print(f"Input shape: {input_shape}")
        print(f"Warmup: {warmup} iterations")
        print(f"Benchmark: {iterations} iterations")

        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        print("Warming up...")
        for _ in range(warmup):
            self.infer(dummy_input)

        # Benchmark
        print("Benchmarking...")
        start = time.time()
        for _ in range(iterations):
            self.infer(dummy_input)
        end = time.time()

        total_time = end - start
        avg_time = total_time / iterations * 1000  # ms
        fps = iterations / total_time

        print(f"\nResults:")
        print(f"  Total time: {total_time:.2f} s")
        print(f"  Average latency: {avg_time:.2f} ms")
        print(f"  Throughput: {fps:.2f} FPS")

        return {
            'avg_latency_ms': avg_time,
            'fps': fps,
            'total_time_s': total_time
        }

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║                                                          ║")
    print("║         YOLOv8 TensorRT Inference Test                  ║")
    print("║                                                          ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    engine_file = 'yolov8n.engine'

    try:
        # Load engine
        inferencer = TensorRTInference(engine_file)

        # Run benchmark
        results = inferencer.benchmark(
            input_shape=(1, 3, 640, 640),
            warmup=10,
            iterations=100
        )

        # Summary
        print("\n╔══════════════════════════════════════════════════════════╗")
        print("║                    Summary                               ║")
        print("╚══════════════════════════════════════════════════════════╝\n")

        print(f"YOLOv8n Performance on Jetson Orin Nano:")
        print(f"  • Latency: {results['avg_latency_ms']:.2f} ms")
        print(f"  • FPS: {results['fps']:.1f}")
        print(f"  • Input: 640x640 RGB")
        print(f"  • Precision: FP16 (if supported)")

        # Estimate for real-time
        target_fps = 30
        if results['fps'] >= target_fps:
            print(f"\n✓ Can achieve {target_fps} FPS real-time processing!")
        else:
            print(f"\n⚠  Current FPS ({results['fps']:.1f}) below {target_fps} FPS target")
            print(f"   Try smaller input size or INT8 precision")

        print()

    except FileNotFoundError:
        print(f"✗ Error: {engine_file} not found")
        print(f"   Run: python3 scripts/setup_yolov8.py")
        return 1
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
