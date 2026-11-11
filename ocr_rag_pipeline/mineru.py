"""Execute the *existing* MinerU 4.x CLI without touching its Python environment."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .utils import LOG


def resolve_mineru_exe(user_exe: str | None = None) -> str:
    if user_exe:
        resolved = shutil.which(user_exe)
        if resolved:
            return resolved
        if Path(user_exe).is_file():
            return str(Path(user_exe).resolve())
        raise FileNotFoundError(f"找不到指定 MinerU 命令：{user_exe}")

    located = shutil.which("mineru-kit")
    if located:
        return located
    py_dir = Path(sys.executable).resolve().parent
    for candidate in (py_dir / "Scripts" / "mineru-kit.exe", py_dir / "Scripts" / "mineru-kit.cmd",
                      py_dir / "mineru-kit.exe"):
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "无法找到 mineru-kit。先在当前 PowerShell 运行 'Get-Command mineru-kit'，"
        "再使用 --mineru-exe 指向它的完整路径；无需重新安装。"
    )


def parse_pdf(pdf: Path, destination_zip: Path, log_file: Path,
              exe: str | None = None, timeout_seconds: int = 1800) -> None:
    """Parse one PDF; CLI ZIP output has markdown.md and images/ internally."""
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        raise ValueError(f"输入必须是存在的 PDF 文件：{pdf}")
    destination_zip.parent.mkdir(parents=True, exist_ok=True)
    executable = resolve_mineru_exe(exe)
    cmd = [executable, "parse", str(pdf.resolve()), "-o", str(destination_zip.resolve()),
           "--tier", "basic", "--ocr-mode", "auto", "--format", "zip"]
    LOG.info("开始 MinerU 解析：%s", pdf)
    LOG.info("MinerU 命令：%s", subprocess.list2cmdline(cmd))
    with log_file.open("w", encoding="utf-8", errors="replace") as output:
        try:
            process = subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT,
                                     timeout=timeout_seconds or None, check=False)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"MinerU 超时（{timeout_seconds} 秒），日志：{log_file}") from exc
    if process.returncode != 0:
        content = log_file.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"MinerU 退出码 {process.returncode}，日志末尾：\n{content[-3000:]}")
    if not destination_zip.is_file() or destination_zip.stat().st_size == 0:
        raise RuntimeError(f"MinerU 已退出，但未生成有效 ZIP：{destination_zip}。详见 {log_file}")
    LOG.info("MinerU 输出：%s", destination_zip)
