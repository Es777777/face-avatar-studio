# Face Avatar Studio

Windows 本地桌面表情捕捉与 3D 头像驱动软件。项目基于
[Google GNM](https://github.com/google/GNM) 和
[MediaPipe Face Landmarker](https://github.com/google-ai-edge/mediapipe)，并提供
可选的 [FaceVerse V2](https://github.com/LizhenWangT/FaceVerse#faceverse-version-2)
52 维 ARKit 驱动和 [FaceVerse v4](https://github.com/LizhenWangT/FaceVerse_v4)
全头参数化后端。

[English README](README.en.md)

## 功能

- 摄像头实时采集人脸关键点、表情和头部姿态
- 默认使用 MediaPipe + GNM，低延迟驱动带完整头部的 3D 头像
- 可选 FaceVerse V2：52 维 ARKit 表情基与 MediaPipe 逐项直连
- 可选 FaceVerse v4：621 维参数回归，包含头部、眼球、牙齿和舌头模型
- 在界面中选择摄像头和麦克风
- 实时预览摄像头画面、头像画面、FPS、检测状态和音量
- 一键开始/停止录制，生成同步视频、音频和表情时间轴
- 录制完成后可用 faster-whisper 识别中文或英文，并按时间段导出
- 将表情参数、时间戳、语音片段和识别文本合并到 CSV

## 运行环境

- Windows 10/11
- Python 3.13（当前开发环境）
- 可用摄像头和麦克风
- CPU 可运行；FaceVerse 在有 CUDA 的环境中速度更好

程序使用 Tk 桌面界面启动，避免把 Qt/VTK 的 DLL 加载错误带到主窗口启动阶段。
VTK 只在头像预览线程中按需加载，渲染异常不会直接卡死主界面。

## 安装

在项目根目录执行：

```powershell
python -m pip install -r requirements.txt
```

如果缺少 GNM 源码：

```powershell
git clone https://github.com/google/GNM.git external_GNM
```

MediaPipe 的 `face_landmarker.task` 会按项目配置下载并缓存到用户缓存目录。

## 启动

推荐双击：

```text
start_face_avatar_studio.bat
```

也可以执行：

```powershell
python launch_face_avatar_studio.py
```

启动后，在“追踪后端”中选择：

- `MediaPipe（默认）`：MediaPipe 直接提供表情参数，GNM 负责头像驱动
- `FaceVerse v4`：MediaPipe 只用于稳定人脸 ROI，FaceVerse ResNet50 回归 621 维参数
- `FaceVerse V2（52维）`：MediaPipe 的 52 个 ARKit 表情分数直接驱动 FaceVerse 表情基

选择后端前请先关闭摄像头。FaceVerse 的首次加载会初始化 PyTorch 和模型，可能需要几秒。

## FaceVerse 模型

FaceVerse 权重不随官方源码仓库直接分发。本项目使用独立的 GitHub LFS 模型仓库：

<https://github.com/Es777777/face-avatar-studio>

需要的文件：

```text
external_FaceVerse_v4/data/faceverse_v4_2.npy
external_FaceVerse_v4/data/faceverse_resnet50.pth
```

当前项目目录已经包含这两个文件。手动重新部署时，可从模型仓库下载后放入上述目录。
程序启动时会检查文件是否存在；缺少权重时不会静默伪装成 FaceVerse，而会显示明确的错误提示。

FaceVerse 代码和模型的版权、许可证及研究引用请以原作者仓库为准：
<https://github.com/LizhenWangT/FaceVerse_v4>

FaceVerse V2 的官方可选模型文件位置：

```text
external_FaceVerse_v2/data/faceverse_simple_v2.npy
```

如果该文件存在，V2 后端会自动使用官方 V2 简化网格；如果暂未下载，程序会使用
V4 全头模型的完整下颌、牙齿和口腔联动基底，并把 52 个 ARKit 通道映射到对应的
FaceVerse 表情语义，界面状态栏会明确显示当前模型来源。V2 原始模型与下载说明见：
<https://github.com/LizhenWangT/FaceVerse#faceverse-version-2>

## 录制输出

每次录制会创建独立会话目录，通常包含：

```text
recording_with_audio.mp4   音画合成视频
video_only.mp4             无音频视频
audio.wav                  原始音频
timeline.csv               每帧表情参数、时间戳和追踪状态
transcript_segments.csv    STT 分段结果
session.json               录制配置和元数据
```

STT 使用 faster-whisper。模型可在首次使用时下载，也可以根据本机环境预先准备模型缓存。

## 目录结构

```text
face_avatar_studio/       桌面程序源码
external_GNM/             Google GNM 外部源码
external_FaceVerse_v2/    FaceVerse V2 可选模型目录
external_FaceVerse_v4/    FaceVerse v4 外部源码和模型
tools/diagnostics/        诊断与回归脚本
artifacts/generated/      测试截图和验证结果
artifacts/references/     参考素材
artifacts/logs/           历史日志
.cache/                   MediaPipe 和开发缓存
```

## 常见问题

### FaceVerse 头像倒立、反向或只显示静止模型

请确认两个权重文件都在 `external_FaceVerse_v4/data/`，然后完全退出软件再重启。
当前适配器会将 FaceVerse 原生坐标转换到桌面预览坐标，并使用实时回归结果更新网格。
可先在界面状态栏确认后端显示为 `FaceVerse v4`，并观察帧号和头像 FPS 是否持续变化。

### FaceVerse V2 显示“52维兼容模式”

这是正常状态，表示 V2 的逐项 ARKit 驱动正在使用 V4 全头兼容网格。将官方
`faceverse_simple_v2.npy` 放入 `external_FaceVerse_v2/data/` 并重启软件后，会自动
切换为官方 V2 网格。

### 首次启动较慢

首次运行会加载 MediaPipe、PyTorch、FaceVerse 或 VTK。启动完成后这些模块会留在进程缓存中。
不要同时打开多个软件实例，也不要在摄像头已经打开时切换追踪后端。

### 头像黑屏或程序无响应

关闭摄像头后重新启动头像预览。如果问题仍在，请查看项目根目录的启动日志和
`artifacts/logs/`，并优先使用软件渲染路径排查显卡驱动问题。

## 打包

执行：

```powershell
.\build_windows_app.ps1
```

生成目录通常为 `dist/FaceAvatarStudio/`。发布可执行版本时，需要同时携带外部源码、
MediaPipe 模型缓存配置和 FaceVerse 权重。V2 模型目录也会一并打包。

## 许可证与致谢

本项目整合了多个上游项目。GNM、MediaPipe、FaceVerse、Sim3DR 及其模型的许可证和
使用限制分别以各自仓库为准。请在分发软件时保留对应的许可证文件和引用信息。
