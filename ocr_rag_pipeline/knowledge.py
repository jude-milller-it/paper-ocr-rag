"""Dify V2 indexing, paper structure and evidence-only record creation.

Used by main.py.  Not an independent command or an OCR engine.
Reads only DIFY_API_KEY, DIFY_V2_ID and optionally DIFY_API_BASE.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID

try:
    import requests
except ImportError as exc:
    raise SystemExit("缺少 requests；请使用原来的 Miniforge base Python。") from exc

IMAGE_RE = re.compile(r"!\[([^]]*)\]\((?:<)?(images/[^\s)>]+)(?:>)?\)")
FIGURE_RE = re.compile(r"^Figure\s+(\d+)\s*:\s*(.*)", re.I | re.S)
TABLE_RE = re.compile(r"^Table\s+(\d+)\s*:\s*(.*)", re.I | re.S)
H2_RE = re.compile(r"^##\s+(.+)")
UUID_IN_URL_RE = re.compile(r"(?:^|/)datasets/([0-9a-fA-F-]{36})(?:[/?#]|$)")
STOP_SECTIONS = re.compile(r"^(?:\d+\.\s*)?REFERENCES\b|^ACKNOWLEDGMENTS\b", re.I)


def env(key: str) -> str:
    value = os.getenv(key, "").strip()
    if value:
        return value
    # Read persisted user vars even when the PyCharm process predates the setting.
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as h:
                value = str(winreg.QueryValueEx(h, key)[0]).strip()
        except (OSError, ImportError):
            pass
    return value


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean(text: str) -> str:
    def replace_image(m: re.Match) -> str:
        return "[原始图片路径：" + m.group(2) + "; 图片本身未上传到 Dify]"
    return IMAGE_RE.sub(replace_image, text).strip()




@dataclass
class Record:
    key: str
    name: str
    content: str
    kind: str


def split_blocks(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"\n[ \t]*\n+", text) if x.strip()]


def split_section(paragraphs: list[str], maximum: int = 2200) -> list[str]:
    """Keep paragraphs and headings together; never create heading-only chunks."""
    groups: list[str] = []
    current = ""
    for para in paragraphs:
        if not para.strip():
            continue
        if current and len(current) + len(para) + 2 > maximum:
            groups.append(current.strip())
            current = ""
        # For extraordinarily long formula paragraphs, let Dify's token safeguard
        # split further; do NOT cut a LaTeX environment in half ourselves.
        current = f"{current}\n\n{para}" if current else para
    if current.strip():
        groups.append(current.strip())
    return groups


def make_records(raw: str, slug: str) -> list[Record]:
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    blocks = split_blocks(raw)
    if not blocks:
        raise ValueError("输入的 Markdown 为空")
    paper_title = re.sub(r"^#\s*", "", blocks[0].split("\n", 1)[0]).strip()
    if len(paper_title) > 250:
        paper_title = paper_title[:250]
    figures: dict[int, str] = {}
    tables: dict[int, str] = {}
    sections: dict[str, list[str]] = {}
    heading = "PAPER HEADER"
    sections[heading] = []
    order: list[str] = [heading]
    i = 1
    while i < len(blocks):
        b = blocks[i]
        h = H2_RE.match(b)
        if h:
            heading = h.group(1).strip()
            if STOP_SECTIONS.search(heading):
                # References and acknowledgements are omitted from the answer index:
                # they can cause 'someone else's contribution' false attribution.
                if "REFERENCES" in heading.upper():
                    break
                heading = "ACKNOWLEDGMENTS"
            if heading not in sections:
                sections[heading] = []
                order.append(heading)
            i += 1
            continue
        # Caption after local image link. Keep figure number, caption, visual
        # description, paper-based evidence, and page together in ONE document.
        if IMAGE_RE.fullmatch(b) and i + 1 < len(blocks) and FIGURE_RE.match(blocks[i + 1]):
            n = int(FIGURE_RE.match(blocks[i + 1]).group(1))
            parts = [b, blocks[i + 1]]
            i += 2
            while i < len(blocks):
                next_b = blocks[i]
                if (next_b.startswith(f"**Figure {n}｜") or
                    next_b.startswith("**视觉模型观察") or
                    next_b.startswith("**论文正文关联证据") or
                    next_b.startswith(">")):
                    parts.append(next_b)
                    i += 1
                else:
                    break
            figures[n] = clean("\n\n".join(parts))
            continue
        # Table caption + entire contiguous Markdown table, never orphan labels.
        mt = TABLE_RE.match(b)
        if mt and i + 1 < len(blocks) and blocks[i + 1].lstrip().startswith("|"):
            n = int(mt.group(1))
            tables[n] = b + "\n\n" + blocks[i + 1]
            i += 2
            continue
        if IMAGE_RE.fullmatch(b):
            # Orphan images: keep a textual path in their original section.
            sections[heading].append(clean(b))
        elif not STOP_SECTIONS.search(heading):
            sections[heading].append(clean(b))
        i += 1

    def para_match(section: str, pattern: str, max_items: int = 3) -> list[str]:
        return [p for p in sections.get(section, []) if re.search(pattern, p, re.I)][:max_items]

    # Retrieval-oriented overview: ONLY verbatim material from the original
    # paper; do not synthesize claims with the VLM or invent measurements.
    overview_parts = [
        "# PAPER OVERVIEW | 核心贡献 | main contributions | research findings",
        f"Paper: {paper_title}",
        "以下是论文原文摘录，不是生成模型自行总结：",
    ]
    if sections.get("ABSTRACT"):
        overview_parts.append("[证据: ABSTRACT]\n" + "\n\n".join(sections["ABSTRACT"][:2]))
    intro = para_match("1. INTRODUCTION", r"in this work|our contribution|as we were conducting", 3)
    # Captured headings are not guaranteed to be numbered identically in every PDF.
    if intro:
        overview_parts.append("[证据: INTRODUCTION]\n" + "\n\n".join(intro))
    if sections.get("4. CONCLUSIONS"):
        overview_parts.append("[证据: CONCLUSIONS]\n" + "\n\n".join(sections["4. CONCLUSIONS"][:2]))
    overview = "\n\n".join(overview_parts)
    records: list[Record] = [
        Record("OVERVIEW", f"{slug}__OVERVIEW.txt", overview, "overview")
    ]
    for num, content in sorted(figures.items()):
        records.append(Record(
            f"FIGURE_{num:02d}", f"{slug}__FIGURE_{num:02d}.txt",
            f"# Figure {num} | 图 {num} | {paper_title}\n"
            f"文献类型：图像/图表；本记录只属于 Figure {num}，不可与其他图表混淆。\n\n{content}",
            "figure"))
    for num, content in sorted(tables.items()):
        records.append(Record(
            f"TABLE_{num:02d}", f"{slug}__TABLE_{num:02d}.txt",
            f"# Table {num} | 表 {num} | {paper_title}\n"
            f"文献类型：表格；本记录只属于 Table {num}，不可与其他表格混淆。\n\n{content}",
            "table"))
    for sec in order:
        paras = sections.get(sec) or []
        if not paras or sec in ("PAPER HEADER", "ACKNOWLEDGMENTS", "Keywords"):
            continue
        for part, body in enumerate(split_section(paras), 1):
            heading_key = re.sub(r"[^A-Za-z0-9]+", "_", sec).strip("_")[:24] or "Section"
            key = f"SECTION_{heading_key}_{part:02d}"
            content = (f"# {paper_title}\n## {sec}\n"
                       f"记录类型：论文正文。引用的既有研究不得写成本文原创贡献。\n\n{body}")
            records.append(Record(key, f"{slug}__{key}.txt", content, "section"))
    if len(figures) != len(set(figures)) or len(tables) != len(set(tables)):
        raise ValueError("检测到重复编号，停止入库以避免覆盖")
    declared_figures = len(re.findall(r"(?m)^\*\*Figure \d+｜", raw))
    declared_tables = len(re.findall(r"(?m)^Table \d+:\s", raw))
    if declared_figures and len(figures) < declared_figures:
        raise ValueError(f"视觉增强图像有 {declared_figures} 处标记，但仅提取 {len(figures)} 处；为防漏图已停止")
    if declared_tables and len(tables) < declared_tables:
        raise ValueError(f"发现 {declared_tables} 张表格但只提取了 {len(tables)} 张；为防丢表已停止")
    if not figures or not tables:
        print("注意：原文的图表结构无法完整识别，请人工检查预览", file=sys.stderr)
    return records


def kb_id(selector: str) -> str:
    selector = selector.strip()
    m = UUID_IN_URL_RE.search(selector)
    if m:
        selector = m.group(1)
    try:
        return str(UUID(selector))
    except ValueError as exc:
        raise ValueError("知识库 ID 必须是 UUID，或浏览器里包含 /datasets/UUID 的地址") from exc


class Dify:
    def __init__(self, base: str, api_key: str):
        base = base.rstrip("/")
        url = urlparse(base)
        if url.scheme not in ("http", "https") or not base.endswith("/v1"):
            raise ValueError("DIFY_API_BASE 格式应为 http://localhost:8080/v1")
        self.base = base
        self.s = requests.Session()
        self.s.headers.update({"Authorization": "Bearer " + api_key, "Accept": "application/json"})

    def req(self, method: str, endpoint: str, *, payload=None, params=None):
        try:
            r = self.s.request(method, self.base + endpoint, json=payload, params=params, timeout=45)
        except requests.RequestException as exc:
            raise RuntimeError(f"Dify 请求失败（POST 结果不确定时先查看知识库再重试）：{exc}") from exc
        if not r.ok:
            raise RuntimeError(f"Dify HTTP {r.status_code} {endpoint}：{r.text[:700]}")
        try:
            out = r.json()
        except ValueError as exc:
            raise RuntimeError(f"Dify 返回非 JSON：{r.text[:300]}") from exc
        if not isinstance(out, dict):
            raise RuntimeError(f"Dify 返回异常：{str(out)[:250]}")
        return out

    def doc_list(self, did: str) -> dict[str, dict]:
        docs: dict[str, dict] = {}
        for page in range(1, 101):
            data = self.req("GET", f"/datasets/{did}/documents", params={"page": page, "limit": 100})
            for item in data.get("data", []):
                if item.get("name"):
                    if item["name"] in docs:
                        raise RuntimeError(f"知识库存在重名文档：{item['name']}")
                    docs[item["name"]] = item
            if not data.get("has_more"):
                return docs
        raise RuntimeError("文档超过100页，为安全起见停止")

    @staticmethod
    def rule():
        # Each document is already a logical unit. Dify's max-token safety
        # still applies, but it should not arbitrarily split a normal table.
        return {"mode": "custom", "rules": {
            "pre_processing_rules": [
                {"id": "remove_extra_spaces", "enabled": False},
                {"id": "remove_urls_emails", "enabled": False},
            ],
            "segmentation": {"separator": "[[OCR_RAG_V2_CUSTOM_BREAK_DO_NOT_USE]]",
                             "max_tokens": 1800, "chunk_overlap": 0},
        }}

    def create(self, did: str, record: Record):
        return self.req("POST", f"/datasets/{did}/document/create-by-text", payload={
            "name": record.name, "text": record.content, "indexing_technique": "high_quality",
            "doc_form": "text_model", "doc_language": "English", "process_rule": self.rule(),
        })

    def update(self, did: str, docid: str, record: Record):
        return self.req("POST", f"/datasets/{did}/documents/{docid}/update-by-text", payload={
            "name": record.name, "text": record.content, "doc_form": "text_model",
            "doc_language": "English", "process_rule": self.rule(),
        })

    def wait(self, did: str, pending: dict[str, str], max_seconds=900):
        deadline = time.monotonic() + max_seconds
        while pending:
            if time.monotonic() >= deadline:
                raise RuntimeError("等待索引超时。任务可能仍在继续，检查 Dify 文档状态即可")
            for name, batch in list(pending.items()):
                out = self.req("GET", f"/datasets/{did}/documents/{batch}/indexing-status")
                entries = out.get("data") or []
                if not entries:
                    continue
                statuses = {str(x.get("indexing_status", "")).lower() for x in entries}
                if statuses <= {"completed"}:
                    print(f"  ✓ {name} 索引完成")
                    pending.pop(name)
                elif statuses & {"error", "failed", "stopped", "disabled"}:
                    raise RuntimeError(f"{name} 索引失败：{entries}")
            if pending:
                print(f"等待索引：剩余 {len(pending)} 个文档", flush=True)
                time.sleep(4)

    def smoke(self, did: str):
        checks = [
            ("Table 1 methods accuracy", "__TABLE_01"),
            ("Figure 3 four layers accuracy", "__FIGURE_03"),
            ("paper main contributions method dataset experimental results", "__OVERVIEW"),
        ]
        print("\n自动召回检查（只检查索引，不代表最终 LLM 一定回答正确）：")
        for question, marker in checks:
            try:
                out = self.req("POST", f"/datasets/{did}/retrieve", payload={
                    "query": question,
                    "retrieval_model": {"search_method": "hybrid_search", "reranking_enable": False,
                                        "top_k": 5, "score_threshold_enabled": False},
                })
                records = out.get("records") or []
                def record_name(x):
                    return str(((x.get("segment") or {}).get("document") or {}).get("name") or
                               (x.get("document") or {}).get("name") or x.get("title") or "")
                ranks = [i for i, x in enumerate(records, 1) if marker in record_name(x)]
                if ranks:
                    print(f"  ✓ {question}：目标文档第 {ranks[0]} 名")
                else:
                    print(f"  ? {question}：前 5 条未命中目标文档；需要继续调检索")
            except RuntimeError as exc:
                print(f"  ? 检索自测未完成：{str(exc)[:150]}")


def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")




def ingest_markdown(
    md: Path, *, prepare_only: bool = False, dataset_id: str | None = None,
    api_base: str | None = None, skip_smoke: bool = False,
) -> dict:
    """Structure ONE explicitly supplied Markdown and index it in Dify.

    Never auto-select a "latest" paper: the caller passes the exact output of OCR.
    Legacy V2 document names and upload state are intentionally preserved, so
    documents already uploaded using rag_v2.py are skipped instead of duplicated.
    """
    md = Path(md)
    if not md.is_file():
        raise FileNotFoundError(f"找不到增强 Markdown：{md}")
    raw = md.read_text(encoding="utf-8-sig")
    if not any(tag in raw for tag in ("图像辅助信息", "图像可见信息补充", "图像辅助解析")):
        raise ValueError(f"文件不像视觉增强后的 Markdown：{md}")
    paper_title = re.sub(r"^#\s*", "", raw.splitlines()[0]).strip()
    safe_title = re.sub(r"[^a-zA-Z0-9]+", "_", paper_title).strip("_")[:38]
    legacy_slug = f"{safe_title or 'paper'}_{digest(paper_title)[:8]}"
    # Previously indexed papers retain their document names and state IDs.
    # Brand-new papers get compact, consistent names instead of long titles.
    # Resolve the remote naming convention BEFORE creating any local files.
    # This prevents a mismatch between the preprocessed document paths and
    # the remote-document names after adopting a previous V2 upload.
    selector = dataset_id or env("DIFY_V2_ID")
    key = env("DIFY_API_KEY")
    did = None
    client = None
    existing = {}
    if not prepare_only:
        if not selector or not key:
            raise ValueError("缺少 DIFY_V2_ID 或 DIFY_API_KEY；请先保存到 Windows 用户环境变量")
        did = kb_id(selector)
        client = Dify(api_base or env("DIFY_API_BASE") or "http://localhost:8080/v1", key)
        details = client.req("GET", f"/datasets/{did}")
        print(f"Dify 知识库：{details.get('name', did)}")
        if details.get("indexing_technique") not in (None, "high_quality"):
            raise ValueError("目标库不是高质量索引")
        if details.get("doc_form") not in (None, "text_model"):
            raise ValueError("目标库不是通用分块模式（text_model）")
        existing = client.doc_list(did)

    state_root = Path("data/output/rag_v2_state")
    legacy_exists = list(state_root.glob(f"*/{legacy_slug}.json"))
    # Old V2 document names take precedence if they exist in Dify, regardless
    # of whether local status files survived earlier refactors.
    remotely_legacy = any(name.startswith(legacy_slug + "__") for name in existing)
    slug = legacy_slug if (legacy_exists or remotely_legacy) else f"paper_{digest(paper_title)[:10]}"
    if remotely_legacy:
        print("发现旧版 V2 文档，采用其原名称及远程文档 ID")
    records = make_records(raw, slug)
    folder = md.parent / "knowledge"
    folder.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": str(md.resolve()), "source_sha256": digest(raw),
        "overview": sum(r.kind == "overview" for r in records),
        "figures": sum(r.kind == "figure" for r in records),
        "tables": sum(r.kind == "table" for r in records),
        "sections": sum(r.kind == "section" for r in records),
        "records": [{"name": r.name, "type": r.kind, "sha256": digest(r.content)} for r in records],
    }
    # Local cache filenames MUST stay short on Windows (MAX_PATH ~260).
    # Dify document names are independent and remain unchanged, preserving
    # matching IDs for all previously uploaded V2 records.
    for index, rec in enumerate(records, 1):
        target = folder / f"part_{index:02d}.txt"
        target.write_text(rec.content, encoding="utf-8")
        if not target.is_file():
            raise FileNotFoundError(f"生成结构化文档失败：{target}")
    save_json(folder / "manifest.json", summary)
    print(f"结构化：{len(records)} 份文档（图 {summary['figures']}、表 {summary['tables']}、"
          f"正文 {summary['sections']}、概览 {summary['overview']}）")
    if prepare_only:
        print("仅预处理；未调用 Dify")
        return {"status": "prepared", **summary}

    foreign = [name for name in existing
               if not re.search(r"(?:_[0-9a-f]{8}|paper_[0-9a-f]{10})__(?:OVERVIEW|FIGURE_|TABLE_|SECTION_)", name)]
    if foreign:
        raise ValueError(f"V2 库含非结构化文档：{foreign[0]}；为了避免污染检索，已停止")

    # Backward-compatible with the state produced by the previously deployed rag_v2.py.
    state_dir = Path("data/output/rag_v2_state") / did
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f"{slug}.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {
        "dataset_id": did, "docs": {},
    }
    pending: dict[str, str] = {}
    counts = {"new": 0, "updated": 0, "skipped": 0}
    for index, rec in enumerate(records, start=1):
        hash_ = digest(rec.content)
        remote = existing.get(rec.name)
        prev = state.get("docs", {}).get(rec.name)
        if remote and not prev:
            # Exact names in this dataset are unique, deterministic, paper-specific
            # identifiers. Rebuild missing state by adopting the remote document ID.
            # Unknown content hashes are deliberately not assumed equal: update the
            # existing document in place, never create a duplicate.
            rid = remote.get("id")
            if not rid:
                raise ValueError(f"Dify 已有同名文档但缺少 ID：{rec.name}")
            prev = {"id": rid, "sha256": None, "status": "adopted"}
            state.setdefault("docs", {})[rec.name] = prev
            save_json(state_path, state)
        if remote and prev and prev.get("id") != remote.get("id"):
            raise ValueError(f"远程文档 ID 已改变，停止：{rec.name}")
        if remote and prev and prev.get("sha256") == hash_ and remote.get("indexing_status") == "completed":
            counts["skipped"] += 1
            continue
        print(f"[{index}/{len(records)}] {'更新' if remote else '上传'} {rec.name}", flush=True)
        response = client.update(did, remote["id"], rec) if remote else client.create(did, rec)
        docid = (response.get("document") or {}).get("id") or (remote or {}).get("id")
        batch = response.get("batch")
        if not docid or not batch:
            raise RuntimeError(f"上传响应缺少文档 ID 或批次，先检查 Dify 再重试：{rec.name}")
        state["docs"][rec.name] = {"id": docid, "sha256": hash_, "batch": str(batch),
                                   "status": "submitted"}
        save_json(state_path, state)
        pending[rec.name] = str(batch)
        counts["updated" if remote else "new"] += 1
    if pending:
        client.wait(did, pending)
        for name in pending:
            state["docs"][name]["status"] = "completed"
        save_json(state_path, state)
    print(f"✅ Dify 入库完成：新建 {counts['new']}、更新 {counts['updated']}、"
          f"跳过已存在 {counts['skipped']}")
    if not skip_smoke:
        client.smoke(did)
    return {"status": "indexed", "counts": counts, "dataset_id": did, **summary}
