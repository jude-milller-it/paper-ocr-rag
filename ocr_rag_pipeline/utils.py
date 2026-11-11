"""Shared filesystem and logging helpers. Python >=3.10; no third-party deps."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

LOG = logging.getLogger("ocr_rag_pipeline")


def safe_stem(stem: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip(" .")
    return (cleaned or "document")[:75]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def setup_logging(log_file: Path) -> None:
    LOG.setLevel(logging.INFO)
    for handler in list(LOG.handlers):
        LOG.removeHandler(handler)
        handler.close()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    disk = logging.FileHandler(log_file, encoding="utf-8")
    disk.setFormatter(fmt)
    LOG.addHandler(console)
    LOG.addHandler(disk)
