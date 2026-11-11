"""Carefully extract MinerU ZIP with protection against path traversal."""
from __future__ import annotations

import shutil
import stat
from pathlib import Path
from zipfile import ZipFile, BadZipFile


class ArchiveError(RuntimeError):
    pass


def unpack_zip(archive: Path, output: Path, max_bytes: int = 2 * 1024**3) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    root = output.resolve()
    total = 0
    try:
        with ZipFile(archive) as zf:
            if len(zf.infolist()) > 10000:
                raise ArchiveError("压缩包文件数量过多")
            for item in zf.infolist():
                # Windows backslashes must also be treated as path separators.
                parts = item.filename.replace("\\", "/").split("/")
                if item.filename.startswith(("/", "\\")) or any(p in ("..", "") for p in parts if p != parts[-1]):
                    raise ArchiveError(f"压缩包含非法路径：{item.filename}")
                if ":" in parts[0] or ".." in parts:
                    raise ArchiveError(f"压缩包含非法路径：{item.filename}")
                unix_mode = item.external_attr >> 16
                if stat.S_ISLNK(unix_mode):
                    raise ArchiveError(f"不允许压缩包中的符号链接：{item.filename}")
                target = output.joinpath(*[p for p in parts if p]).resolve()
                if target != root and root not in target.parents:
                    raise ArchiveError(f"压缩包路径越界：{item.filename}")
                total += item.file_size
                if total > max_bytes:
                    raise ArchiveError("解压数据总量超出 2 GiB 上限")
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(item, "r") as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
    except BadZipFile as exc:
        raise ArchiveError(f"不是有效的 MinerU ZIP：{archive}") from exc
    matches = list(output.rglob("markdown.md"))
    if len(matches) != 1:
        raise ArchiveError(f"预期恰好找到一个 markdown.md，实际 {len(matches)} 个")
    return matches[0].parent
