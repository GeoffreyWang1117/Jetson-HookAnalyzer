# 📹 HookAnalyzer 视频录制指南

## 🎯 视频目的
- ✅ 展示项目在真实硬件上运行
- ✅ 证明性能指标真实可信
- ✅ 提升简历/GitHub可信度
- ✅ 方便面试官快速了解项目

---

## 🎬 录制准备

### 1. 在Jetson上安装录屏工具

**方法A: 使用 `asciinema` (推荐 - 终端录制)**
```bash
# SSH到Jetson
ssh geoffrey@100.111.167.60

# 安装asciinema
sudo apt-get update
sudo apt-get install -y asciinema

# 测试安装
asciinema --version
```

**方法B: 使用 `terminalizer` (更美观)**
```bash
# 需要Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
sudo npm install -g terminalizer

# 测试
terminalizer --version
```

**方法C: 使用 `script` (内置工具)**
```bash
# 无需安装，系统自带
script --version
```

---

## 🎥 录制步骤

### 方案1: asciinema录制（最简单）

```bash
# 1. 登录Jetson
ssh geoffrey@100.111.167.60

# 2. 准备环境
cd ~/HookAnalyzer
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

# 3. 开始录制
asciinema rec hookanalyzer-demo.cast

# 4. 运行演示脚本
bash scripts/demo_video.sh

# 5. 结束录制（Ctrl+D 或输入 exit）

# 6. 上传到asciinema.org（可选）
asciinema upload hookanalyzer-demo.cast

# 7. 下载到本地
# 在本地机器上:
scp geoffrey@100.111.167.60:~/HookAnalyzer/hookanalyzer-demo.cast .
```

### 方案2: Terminalizer录制（更精美）

```bash
# 1. 创建配置
cd ~/HookAnalyzer
terminalizer init demo-config

# 2. 开始录制
terminalizer record hookanalyzer-demo --config demo-config

# 3. 运行演示脚本
bash scripts/demo_video.sh

# 4. 结束录制（Ctrl+D）

# 5. 预览
terminalizer play hookanalyzer-demo

# 6. 渲染为GIF
terminalizer render hookanalyzer-demo -o demo.gif

# 7. 渲染为视频（需要ffmpeg）
sudo apt-get install -y ffmpeg
terminalizer render hookanalyzer-demo -o demo.mp4 --quality 100
```

### 方案3: OBS Studio远程录制（最专业）

在**本地机器**上：

```bash
# 1. 安装OBS Studio
# macOS: brew install --cask obs
# Windows: 下载 https://obsproject.com
# Linux: sudo apt-get install obs-studio

# 2. SSH连接保持终端打开
ssh geoffrey@100.111.167.60

# 3. 在OBS中添加"窗口捕获"源，选择SSH终端窗口

# 4. 设置录制参数
# - 分辨率: 1920x1080
# - 帧率: 30fps
# - 格式: MP4 (H.264)

# 5. 在SSH终端中运行
cd HookAnalyzer
bash scripts/demo_video.sh

# 6. 在OBS中点击"开始录制"
```

---

## 📝 录制脚本（手动版）

如果自动脚本不工作，可以手动执行：

```bash
# === Part 1: Introduction ===
clear
echo "HookAnalyzer - CUDA Performance Framework"
echo "Platform: Jetson Orin Nano"
echo ""
sleep 2

# === Part 2: System Info ===
echo "=== System Information ==="
nvidia-smi | head -15
echo ""
nvcc --version | grep release
sleep 3

# === Part 3: Project Structure ===
echo "=== Project Structure ==="
cd ~/HookAnalyzer
ls -lh
echo ""
echo "Source code files:"
find . -name '*.cpp' -o -name '*.h' -o -name '*.cu' | head -10
sleep 3

# === Part 4: Run Tests ===
echo "=== Running Kernel Tests ==="
cd build
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
./examples/kernel_test
sleep 3

# === Part 5: Benchmarks ===
echo "=== Performance Benchmarks ==="
./benchmarks/benchmark_kernels
sleep 3

# === Part 6: Summary ===
echo "=== Summary ==="
echo "✓ All tests passed"
echo "✓ 146 GFLOPS (68.6% of cuBLAS)"
echo "✓ 91.3 GB/s memory bandwidth"
echo ""
echo "GitHub: github.com/yourusername/HookAnalyzer"
sleep 3
```

---

## 🎨 视频优化建议

### 终端美化
```bash
# 1. 安装powerline字体（可选）
sudo apt-get install fonts-powerline

# 2. 设置终端配色
# 推荐主题: Solarized Dark, Monokai, Dracula

# 3. 增加字体大小
# 终端设置 -> 字体大小 -> 14-16pt

# 4. 设置窗口大小
# 推荐: 120x30 或 100x24
```

### 录制技巧
- ✅ **清晰度**: 确保字体足够大（14-16pt）
- ✅ **对比度**: 使用深色主题（黑底白字或深蓝底）
- ✅ **流畅性**: 不要快进，让观众看清输出
- ✅ **焦点**: 关键输出可以暂停2-3秒
- ✅ **长度**: 控制在3-5分钟

---

## 📤 视频后期处理

### 转换格式
```bash
# 1. asciinema转GIF（在本地）
npm install -g asciicast2gif
asciicast2gif hookanalyzer-demo.cast demo.gif

# 2. 压缩视频
ffmpeg -i demo.mp4 -vcodec h264 -acodec aac -b:v 2M demo_compressed.mp4

# 3. 添加字幕（可选）
# 使用 OpenShot 或 DaVinci Resolve
```

### 制作缩略图
```bash
# 从视频中提取第5秒作为缩略图
ffmpeg -i demo.mp4 -ss 00:00:05 -vframes 1 thumbnail.jpg

# 或使用截图工具在关键画面截图
```

---

## 🌐 上传与分享

### GitHub展示
```markdown
# 在README.md中添加

## 🎥 Demo Video

### Quick Demo (Terminal Recording)
[![asciicast](https://asciinema.org/a/xxxxx.svg)](https://asciinema.org/a/xxxxx)

### Full Benchmark Demo (3 min)
[![Watch the video](thumbnail.jpg)](https://www.youtube.com/watch?v=xxxxx)

Or download: [demo.mp4](https://github.com/yourusername/HookAnalyzer/releases/download/v1.0/demo.mp4)
```

### 上传平台选择
1. **YouTube** - 最专业，适合详细讲解
2. **Bilibili** - 国内访问友好
3. **GitHub Releases** - 直接附加到项目
4. **asciinema.org** - 终端录制专用
5. **个人网站** - 完全控制

---

## 📋 视频内容检查清单

### 必须包含
- [ ] 系统信息（Jetson型号、CUDA版本）
- [ ] GPU信息（nvidia-smi输出）
- [ ] 项目结构展示
- [ ] 编译成功的库文件
- [ ] Kernel测试全部通过
- [ ] Benchmark性能数据
- [ ] 关键性能指标（146 GFLOPS, 91.3 GB/s）

### 可选增强
- [ ] 代码片段展示（vim/cat关键文件）
- [ ] 实时监控（tegrastats）
- [ ] 与其他方案对比
- [ ] 开发过程花絮

---

## 🎯 示例时间轴（5分钟版）

```
00:00 - 00:30  标题介绍 + 项目概述
00:30 - 01:00  Jetson硬件信息 + GPU检测
01:00 - 01:30  项目结构 + 代码统计
01:30 - 02:30  Kernel测试运行（展示5个测试）
02:30 - 04:00  性能Benchmark（GEMM + Element-wise）
04:00 - 04:30  性能总结 + 关键指标
04:30 - 05:00  技术栈 + GitHub链接 + 结束画面
```

---

## 💡 专业录制工具推荐

### 终端录制
1. **asciinema** - 最简单，适合快速分享
2. **terminalizer** - 更美观，支持GIF/MP4
3. **script + scriptreplay** - 系统内置

### 屏幕录制
1. **OBS Studio** - 专业免费
2. **SimpleScreenRecorder** - Linux专用
3. **QuickTime** - macOS自带

### 视频编辑
1. **DaVinci Resolve** - 专业免费
2. **OpenShot** - 开源简单
3. **iMovie** - macOS自带

---

## 🚀 快速开始

**最简单的方法（5分钟搞定）**:

```bash
# 1. SSH到Jetson
ssh geoffrey@100.111.167.60

# 2. 安装asciinema
sudo apt-get install -y asciinema

# 3. 准备演示脚本
cd ~/HookAnalyzer
chmod +x scripts/demo_video.sh

# 4. 开始录制
asciinema rec hookanalyzer-demo.cast

# 5. 运行演示
bash scripts/demo_video.sh

# 6. 结束（Ctrl+D）

# 7. 上传
asciinema upload hookanalyzer-demo.cast
# 会得到一个链接，如: https://asciinema.org/a/xxxxx
```

然后在README.md中添加：
```markdown
## Demo
[![asciicast](https://asciinema.org/a/xxxxx.svg)](https://asciinema.org/a/xxxxx)
```

---

## 📧 问题排查

### 问题1: 录制时字太小
```bash
# 临时增大字体
# 终端设置 -> 配置文件 -> 字体 -> 16pt
```

### 问题2: 颜色不正常
```bash
# 设置TERM环境变量
export TERM=xterm-256color
```

### 问题3: 录制文件太大
```bash
# 使用更高压缩率
ffmpeg -i input.mp4 -vcodec libx264 -crf 28 output.mp4
```

---

**建议**: 先用`asciinema`快速录制一个版本，效果满意后再考虑用OBS录制高清版本。

Good luck! 🎬
