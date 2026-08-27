from __future__ import annotations

import ctypes
import os
import sys
import traceback
from pathlib import Path


# 必须在 import 任何重模块之前把这些环境变量和日志等级钉死，
# 否则 mediapipe / tensorflow 启动时会把 absl / oneDNN / XNNPACK 的 C++ 警告
# 全部刷到 stderr，淹没人脸跟踪运行时和摄像头的真实日志。
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_LOG_LEVEL", "3")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    import warnings as _warnings
    _warnings.filterwarnings("ignore")
except Exception:
    pass


def _isolate_dll_search_path() -> None:
    """conda 环境里的 Library/bin 自带 Qt5*_conda.dll，会污染 PySide6 的 DLL 加载。
    通过 SetDefaultDllDirectories + AddDllDirectory 显式指定合法搜索路径，
    把 conda 的 Library/bin 排除掉。
    """

    # 1. 清掉进程 PATH 里的 conda Library/bin（避免某些 loader 还看 PATH）
    env_path = os.environ.get("PATH", "")
    cleaned = []
    for p in env_path.split(os.pathsep):
        norm = p.lower().replace("/", "\\")
        if "miniconda3\\library\\bin" in norm or "anaconda3\\library\\bin" in norm:
            continue
        cleaned.append(p)
    if cleaned:
        os.environ["PATH"] = os.pathsep.join(cleaned)

    # 2. 设置默认 DLL 搜索目录集合
    try:
        k32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        k32.SetDefaultDllDirectories.argtypes = [ctypes.c_uint]
        k32.SetDefaultDllDirectories.restype = ctypes.c_bool
        k32.AddDllDirectory.argtypes = [ctypes.c_wchar_p]
        k32.AddDllDirectory.restype = ctypes.c_void_p

        # LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000
        k32.SetDefaultDllDirectories(0x00001000)

        candidates = [
            r"C:\Windows\System32",
            r"C:\Windows\System32\Wbem",
            r"C:\Windows\System32\WindowsPowerShell\v1.0",
        ]

        py_dir = Path(sys.prefix) / "Lib" / "site-packages" / "PySide6"
        if py_dir.is_dir():
            candidates.append(str(py_dir))
            candidates.append(str(py_dir / "plugins"))
            candidates.append(str(py_dir / "plugins" / "platforms"))
            # Qt 自己的 QCoreApplication::libraryPaths() 也是从环境变量拿
            # 显式覆盖，避免 conda 的 Library/plugins 抢先
            os.environ["QT_PLUGIN_PATH"] = str(py_dir / "plugins")
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(py_dir / "plugins" / "platforms")

        pyhome_bin = Path(sys.prefix) / "DLLs"
        if pyhome_bin.is_dir():
            candidates.append(str(pyhome_bin))

        for c in candidates:
            try:
                k32.AddDllDirectory(c)
            except OSError:
                pass
    except OSError:
        # 非 Windows 或权限受限 → 走原始路径
        pass


def _show_error_dialog(message: str) -> None:
    try:
        ctypes.windll.user32.MessageBoxW(0, message, "Face Avatar Studio 启动失败", 0x00000010)
    except Exception:
        pass


def main() -> int:
    try:
        # 必须在 import 任何 PySide6 之前调用
        from face_avatar_studio.tk_ui import main as run_tk_app

        return run_tk_app()
    except Exception as exc:
        error_text = "".join(traceback.format_exception(exc))
        error_log = Path.cwd() / "FaceAvatarStudio_launch_error.log"
        try:
            error_log.write_text(error_text, encoding="utf-8")
        except Exception:
            pass
        print("Failed to start Face Avatar Studio.")
        print(error_text)
        print(f"Launch error log: {error_log}")
        _show_error_dialog(f"程序启动失败。\n\n{exc}\n\n日志：{error_log}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
