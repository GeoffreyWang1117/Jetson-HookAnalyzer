#include <iostream>
#include <vector>
#include "../engines/tensorrt_adapter/tensorrt_engine.h"

int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                                          ║\n";
    std::cout << "║         TensorRT Engine Test - YOLOv8n                  ║\n";
    std::cout << "║                                                          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

    std::string engine_path = "yolov8n.engine";

    if (argc > 1) {
        engine_path = argv[1];
    }

    try {
        // Load engine
        std::cout << "=== Step 1: Loading TensorRT Engine ===\n";
        hook_analyzer::tensorrt::TensorRTEngine engine(engine_path);

        std::cout << "\n=== Step 2: Engine Information ===\n";
        std::cout << "Number of inputs: " << engine.getNumInputs() << "\n";
        std::cout << "Number of outputs: " << engine.getNumOutputs() << "\n";

        auto input_shape = engine.getInputShape(0);
        std::cout << "Input shape: [";
        for (size_t i = 0; i < input_shape.size(); i++) {
            std::cout << input_shape[i];
            if (i < input_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";

        auto output_shape = engine.getOutputShape(0);
        std::cout << "Output shape: [";
        for (size_t i = 0; i < output_shape.size(); i++) {
            std::cout << output_shape[i];
            if (i < output_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";

        // Run benchmark
        std::cout << "\n=== Step 3: Running Benchmark ===\n";
        auto stats = engine.benchmark(10, 100);

        // Summary
        std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
        std::cout << "║                    Summary                               ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

        float fps = 1000.0f / stats.avg_latency_ms;

        std::cout << "Performance Metrics:\n";
        std::cout << "  • Average Latency: " << stats.avg_latency_ms << " ms\n";
        std::cout << "  • Min Latency: " << stats.min_latency_ms << " ms\n";
        std::cout << "  • Max Latency: " << stats.max_latency_ms << " ms\n";
        std::cout << "  • Throughput: " << fps << " FPS\n\n";

        // Real-time capability check
        if (fps >= 30.0f) {
            std::cout << "✓ Real-time capable (>= 30 FPS)\n";
        } else if (fps >= 20.0f) {
            std::cout << "⚠  Near real-time (" << fps << " FPS)\n";
        } else {
            std::cout << "✗ Below real-time (" << fps << " FPS)\n";
        }

        std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
        std::cout << "║              Test Completed Successfully!                ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n✗ Error: " << e.what() << "\n\n";
        std::cerr << "Make sure:\n";
        std::cerr << "  1. yolov8n.engine exists in the current directory\n";
        std::cerr << "  2. TensorRT libraries are properly installed\n";
        std::cerr << "  3. CUDA runtime is available\n\n";
        return 1;
    }
}
