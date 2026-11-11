"""Best-effort, evidence-grounded PDF page and block-type matching.

MinerU 4.x has pages[].page_idx (0-based). Do not blindly interpret figure
numbers as PDF page numbers. Missing metadata is explicitly represented.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

from .utils import LOG, read_json

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
SUPPORTED_VL_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def normalize_reference(ref: str) -> str:
    ref = unquote(ref.strip().strip('"\''))
    return ref.removeprefix("./").replace("\\", "/")


def references_in_string(text: str) -> set[str]:
    candidates: set[str] = set()
    # Names emitted by MinerU are whitespace-free; image folder is authoritative.
    candidates.update(re.findall(r"images/[A-Za-z0-9_./%\-]+\.(?:jpe?g|png|webp|bmp|gif)", text, re.I))
    return {normalize_reference(p) for p in candidates}


def block_kind(type_name: str) -> str:
    label = type_name.lower()
    if "equation" in label or "formula" in label:
        return "equation"
    if "table" in label:
        return "table"
    if "chart" in label or "plot" in label or "graph" in label:
        return "chart"
    return "image"


def lookup_image_metadata(parsed_dir: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for file in ("structured_content.json", "middle_json.json"):
        path = parsed_dir / file
        if not path.is_file():
            continue
        try:
            payload = read_json(path)
        except (ValueError, OSError) as exc:
            LOG.warning("结构化文件读取失败：%s（%s）", path, exc)
            continue
        if not isinstance(payload, dict):
            continue
        pages = payload.get("pages")
        if not isinstance(pages, list):
            # Old schema compat only; new 4.0 uses pages.
            pages = payload.get("pdf_info", [])
        for page in pages:
            if not isinstance(page, dict):
                continue
            idx = page.get("page_idx")
            page_no = idx + 1 if isinstance(idx, int) and idx >= 0 else None

            def walk(value: object, parent_type: str = "") -> None:
                if isinstance(value, list):
                    for child in value:
                        walk(child, parent_type)
                    return
                if not isinstance(value, dict):
                    return
                this_type = str(value.get("type", parent_type)).lower()
                paths: set[str] = set()
                for key, item in value.items():
                    if isinstance(item, str) and key.lower() in (
                        "img_path", "image_path", "src", "path", "image", "content", "html"
                    ):
                        paths |= references_in_string(item)
                for ref in paths:
                    kind = block_kind(this_type)
                    existing = records.get(ref)
                    # First source (structured_content.json) has priority.
                    if existing is None:
                        records[ref] = {"page": page_no, "kind": kind, "source": file}
                    elif existing["page"] is None and page_no is not None:
                        existing["page"] = page_no
                    elif existing["kind"] == "image" and kind != "image":
                        existing["kind"] = kind
                for child in value.values():
                    if isinstance(child, (dict, list)):
                        walk(child, this_type)
            walk(page.get("blocks", page), "")
    return records


def inferred_metadata(reference: str, known: dict[str, dict]) -> dict:
    ref = normalize_reference(reference)
    info = dict(known.get(ref, {}))
    filename = Path(ref).name
    if info.get("page") is None:
        # Official 4.x ZIP naming: page_{page_idx}_{block_kind}_... (0 based)
        matched = re.match(r"^page_(\d+)_(image|chart|table|equation)_", filename, re.I)
        if matched:
            info["page"] = int(matched.group(1)) + 1
            info["page_source"] = "MinerU ZIP filename (zero-based)"
    if not info.get("kind") or info["kind"] == "image":
        match = re.match(r"^page_\d+_(image|chart|table|equation)_", filename, re.I)
        info["kind"] = match.group(1).lower() if match else info.get("kind", "image")
    return info
