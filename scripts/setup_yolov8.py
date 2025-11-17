#!/usr/bin/env python3
"""
Setup script to download YOLOv8 and convert to TensorRT engine
"""

import os
import sys
import subprocess
import urllib.request

def check_requirements():
    """Check if required packages are installed"""
    requirements = {
        'tensorrt': 'TensorRT',
        'numpy': 'NumPy',
        'onnx': 'ONNX'
    }

    missing = []
    for module, name in requirements.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            missing.append(module)
            print(f"✗ {name} not found")

    return missing

def install_ultralytics():
    """Install ultralytics for YOLOv8"""
    print("\n=== Installing ultralytics ===")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "--user"])
        print("✓ ultralytics installed")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install ultralytics: {e}")
        return False
    return True

def export_yolov8_to_onnx():
    """Export YOLOv8n to ONNX format"""
    print("\n=== Exporting YOLOv8n to ONNX ===")

    script = """
from ultralytics import YOLO
import sys

try:
    # Load YOLOv8n model (smallest/fastest)
    model = YOLO('yolov8n.pt')

    # Export to ONNX
    model.export(
        format='onnx',
        imgsz=640,
        simplify=True,
        opset=17,
        dynamic=False  # Fixed batch size for TensorRT
    )

    print("✓ YOLOv8n exported to ONNX successfully")
    sys.exit(0)
except Exception as e:
    print(f"✗ Export failed: {e}")
    sys.exit(1)
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("✗ Export timeout (5 minutes)")
        return False
    except Exception as e:
        print(f"✗ Export error: {e}")
        return False

def convert_onnx_to_tensorrt():
    """Convert ONNX model to TensorRT engine"""
    print("\n=== Converting ONNX to TensorRT ===")

    script = """
import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print(f"Loading ONNX file: {onnx_path}")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse ONNX file')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    print(f"Network inputs: {network.num_inputs}")
    print(f"Network outputs: {network.num_outputs}")

    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # Enable FP16 if available
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ FP16 mode enabled")

    # Build engine
    print("Building TensorRT engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("ERROR: Failed to build engine")
        return None

    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"✓ Engine saved to {engine_path}")
    return True

# Build engine
onnx_file = 'yolov8n.onnx'
engine_file = 'yolov8n.engine'

if os.path.exists(onnx_file):
    build_engine(onnx_file, engine_file)
else:
    print(f"ERROR: {onnx_file} not found")
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes for engine building
        )

        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            return False

        return True
    except subprocess.TimeoutExpired:
        print("✗ TensorRT build timeout (10 minutes)")
        return False
    except Exception as e:
        print(f"✗ Conversion error: {e}")
        return False

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║                                                          ║")
    print("║         YOLOv8 Setup for HookAnalyzer                   ║")
    print("║                                                          ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # Step 1: Check requirements
    print("=== Step 1: Checking requirements ===")
    missing = check_requirements()

    if 'onnx' in missing:
        print("\nInstalling ONNX...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx", "--user"])

    # Step 2: Install ultralytics
    print("\n=== Step 2: Installing ultralytics ===")
    if not install_ultralytics():
        print("\n✗ Setup failed at ultralytics installation")
        return 1

    # Step 3: Export to ONNX
    print("\n=== Step 3: Exporting YOLOv8n to ONNX ===")
    if not export_yolov8_to_onnx():
        print("\n✗ Setup failed at ONNX export")
        return 1

    # Step 4: Convert to TensorRT
    print("\n=== Step 4: Converting to TensorRT ===")
    if not convert_onnx_to_tensorrt():
        print("\n✗ Setup failed at TensorRT conversion")
        return 1

    # Success
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║                                                          ║")
    print("║              ✓ Setup completed successfully!             ║")
    print("║                                                          ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    print("Generated files:")
    print("  • yolov8n.pt      - PyTorch weights")
    print("  • yolov8n.onnx    - ONNX model")
    print("  • yolov8n.engine  - TensorRT engine\n")

    print("Next steps:")
    print("  1. Test the engine with: python3 test_yolov8_inference.py")
    print("  2. Integrate with scheduler")
    print("  3. Run multi-model benchmarks\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
