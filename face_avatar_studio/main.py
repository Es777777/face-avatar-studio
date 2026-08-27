from __future__ import annotations

# 这个文件是 .exe 打包时的入口（PyInstaller --onefile/--onedir）
# 内部仍走 .__main__ 的 main()，那里的 DLL 隔离逻辑会自动生效。

from .__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())