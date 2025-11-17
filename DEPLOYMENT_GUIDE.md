# HookAnalyzer 部署指南

## 📋 部署清单

### 开发环境 (本地 x86_64)
- [x] Jetson设备连接测试 (100.111.167.60)
- [x] 项目框架搭建完成
- [x] Docker配置文件就绪
- [ ] 本地Docker构建测试
- [ ] 代码推送到Jetson
- [ ] Jetson上编译测试

### 生产环境 (Jetson Orin Nano)
- [ ] JetPack 5.1+ 安装确认
- [ ] TensorRT 8.x 安装确认
- [ ] 系统服务配置
- [ ] 监控系统部署
- [ ] 性能基准测试

---

## 🚀 快速部署流程

### Step 1: 本地开发测试

```bash
# 1. 进入项目目录
cd /home/coder-gw/Projects/JetsonProj/HookAnalyzer

# 2. 使用Docker进行本地测试（推荐）
./scripts/run_docker_dev.sh

# 容器内执行:
cd /workspace
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
./examples/simple_demo
```

### Step 2: 部署到Jetson

```bash
# 自动化部署脚本
./scripts/deploy_to_jetson.sh 100.111.167.60

# 或手动部署:
# 1. 同步代码
rsync -avz --exclude 'build/' \
    ./ geoffrey@100.111.167.60:~/HookAnalyzer/

# 2. SSH到Jetson
ssh geoffrey@100.111.167.60

# 3. 编译
cd HookAnalyzer
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_TENSORRT=ON \
      -DENABLE_PROFILING=ON \
      ..
make -j6  # Orin Nano 6核心

# 4. 运行测试
ctest --output-on-failure

# 5. 运行Demo
./examples/simple_demo
```

### Step 3: 启动服务

```bash
# 在Jetson上
cd ~/HookAnalyzer

# 安装Python依赖
pip3 install -r requirements.txt

# 启动API服务器
python3 api/server/main.py

# 或使用systemd服务
sudo systemctl start hookanalyzer
sudo systemctl enable hookanalyzer
```

### Step 4: 验证部署

```bash
# 1. 检查API健康状态
curl http://100.111.167.60:8000/health

# 2. 获取系统统计
curl http://100.111.167.60:8000/system/stats

# 3. 查看Prometheus指标
curl http://100.111.167.60:8000/metrics

# 4. 提交测试推理请求
curl -X POST http://100.111.167.60:8000/inference \
  -H "Content-Type: application/json" \
  -d '{"model_id": "test", "input_data": [1.0, 2.0, 3.0], "priority": 1}'
```

---

## 🐳 Docker部署方案

### 方案A: 开发容器 (本地x86_64)

```bash
# 构建镜像
docker build -t hookanalyzer:dev \
  -f docker/Dockerfile.local .

# 运行容器
docker run --rm -it \
  --gpus all \
  -v $(pwd):/workspace \
  -p 8000:8000 \
  hookanalyzer:dev
```

### 方案B: Jetson容器

```bash
# 在Jetson上构建
docker build -t hookanalyzer:jetson-dev \
  -f docker/Dockerfile \
  --target jetson-dev \
  .

# 运行
docker run --rm -it \
  --runtime nvidia \
  -v $(pwd):/workspace \
  -v /dev:/dev \
  --privileged \
  -p 8000:8000 \
  hookanalyzer:jetson-dev
```

### 方案C: 完整监控栈

```bash
# 启动所有服务 (API + Prometheus + Grafana)
docker-compose -f docker/docker-compose.yml up -d

# 访问:
# - API: http://100.111.167.60:8000
# - Prometheus: http://100.111.167.60:9090
# - Grafana: http://100.111.167.60:3000 (admin/admin)
```

---

## ⚙️ 配置优化

### Jetson Orin Nano性能模式

```bash
# 设置最大性能模式
sudo nvpmodel -m 0

# 设置风扇为最大转速
sudo jetson_clocks

# 查看当前状态
sudo tegrastats
```

### CUDA环境变量

```bash
# 添加到 ~/.bashrc
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# TensorRT路径 (如果需要)
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
```

### 调度器配置

编辑配置文件或通过API更新:

```json
{
  "num_worker_threads": 4,
  "max_queue_size": 100,
  "enable_dynamic_batching": true,
  "max_batch_size": 8,
  "batch_timeout_ms": 10,
  "num_cuda_streams": 4
}
```

---

## 🔍 故障排查

### 问题1: CUDA库找不到

```bash
# 症状: error while loading shared libraries: libcudart.so
# 解决:
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
sudo ldconfig
```

### 问题2: TensorRT编译失败

```bash
# 症状: NvInfer.h: No such file or directory
# 解决:
sudo apt-get install tensorrt
# 或者禁用TensorRT
cmake -DENABLE_TENSORRT=OFF ..
```

### 问题3: GPU权限问题

```bash
# 症状: CUDA error: no CUDA-capable device is detected
# 解决:
sudo usermod -aG video $USER
# 重新登录
```

### 问题4: Docker容器无法访问GPU

```bash
# 确认NVIDIA Container Runtime已安装
docker run --rm --gpus all nvidia/cuda:11.4.0-base-ubuntu20.04 nvidia-smi

# 如果失败，安装nvidia-docker2
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

---

## 📊 性能监控

### 实时监控脚本

创建 `monitor.sh`:

```bash
#!/bin/bash
while true; do
    clear
    echo "=== HookAnalyzer System Monitor ==="
    echo ""

    # GPU状态
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits

    echo ""
    echo "=== API Stats ==="
    curl -s http://localhost:8000/scheduler/stats | jq '.'

    sleep 2
done
```

### Grafana仪表板

1. 访问 http://100.111.167.60:3000
2. 添加Prometheus数据源 (http://prometheus:9090)
3. 导入仪表板 `monitoring/dashboard/hookanalyzer.json`

---

## 🧪 性能基准测试

```bash
# 运行完整benchmark套件
cd build
./benchmarks/benchmark_kernels

# 输出示例:
# GEMM (512x512x512): 180 GFLOPS (85% of cuBLAS)
# Element-wise Add: 245 GB/s
# ReLU: 198 GB/s
```

---

## 🔐 安全加固

### 生产环境建议

1. **API认证**
   ```python
   # 在 api/server/main.py 添加
   from fastapi.security import HTTPBearer
   security = HTTPBearer()
   ```

2. **HTTPS配置**
   ```bash
   # 使用nginx反向代理
   sudo apt-get install nginx
   # 配置SSL证书
   ```

3. **防火墙规则**
   ```bash
   sudo ufw allow 22/tcp   # SSH
   sudo ufw allow 8000/tcp # API
   sudo ufw enable
   ```

---

## 📝 下一步

- [ ] 集成真实的YOLOv8/ResNet模型
- [ ] 添加模型热加载功能
- [ ] 实现分布式推理（多Jetson）
- [ ] 优化批处理策略
- [ ] 添加模型量化支持

---

## 📞 联系方式

**设备信息**:
- IP: 100.111.167.60
- Hostname: geoffrey-jetson0.tail4c07f3.ts.net
- User: geoffrey
- Platform: Jetson Orin Nano

**项目仓库**: (待创建GitHub repo)
**问题反馈**: GitHub Issues

---

**最后更新**: 2024-11-16
**维护者**: Geoffrey
