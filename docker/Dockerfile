# Multi-stage Dockerfile for HookAnalyzer
# Stage 1: Development image for x86_64 (local testing)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS dev-x86

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    numpy \
    fastapi \
    uvicorn \
    pydantic \
    prometheus-client \
    opencv-python-headless

WORKDIR /app

# Copy source code
COPY . /app/

# Build HookAnalyzer
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_TESTS=OFF \
          -DENABLE_TENSORRT=OFF \
          .. && \
    make -j$(nproc)

# Stage 2: Jetson runtime image (ARM64)
FROM nvcr.io/nvidia/l4t-base:r35.2.1 AS jetson-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    numpy \
    fastapi \
    uvicorn[standard] \
    pydantic \
    prometheus-client

WORKDIR /app

# Copy built artifacts from dev stage
COPY --from=dev-x86 /app/build/lib*.so /app/lib/
COPY --from=dev-x86 /app/api /app/api/
COPY --from=dev-x86 /app/examples /app/examples/

# Add library path
ENV LD_LIBRARY_PATH=/app/lib:${LD_LIBRARY_PATH}

# Expose API port
EXPOSE 8000

# Default command
CMD ["python3", "api/server/main.py"]

# Stage 3: Jetson development image
FROM nvcr.io/nvidia/l4t-tensorrt:r8.5.2-devel AS jetson-dev

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install development dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    python3-dev \
    python3-pip \
    libssl-dev \
    pkg-config \
    vim \
    htop \
    nvtop \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    numpy \
    fastapi \
    uvicorn[standard] \
    pydantic \
    prometheus-client \
    opencv-python \
    matplotlib \
    jupyter

WORKDIR /workspace

# Copy source code
COPY . /workspace/

# Set up build directory
RUN mkdir -p build

# Default command for development
CMD ["/bin/bash"]
