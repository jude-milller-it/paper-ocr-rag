"""OCR-RAG 一键入口: PDF -> MinerU -> Ollama Vision -> structured Dify dataset.

No arguments: process all PDFs in data/input, reuse completed OCR results.
    python main.py
Specific source: python main.py --pdf "path/to/paper.pdf"
Legacy MinerU archive: python main.py --from-zip "path/to/export.zip"
Re-index existing: python main.py --md "path/to/enhanced_markdown.md"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from ocr_rag_pipeline.pipeline import run as run_ocr, find_reusable_archive
from ocr_rag_pipeline.knowledge import ingest_markdown

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "input"
DEFAULT_OUTPUT = ROOT / "data" / "output" / "pipeline"


def find_completed_pdf(pdf: Path, output_root: Path, *, model: str | None = None, language: str | None = None) -> Path | None:
    """Reuse OCR only for byte-identical PDFs with a complete vision report."""
    content_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
    reports = sorted(output_root.glob("*/visual_analysis.json"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    for path in reports:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        stats = record.get("statistics") or {}
        candidate = path.parent / "parsed" / "enhanced_markdown.md"
        if (record.get("pdf_sha256") == content_hash and record.get("status") == "complete"
                and stats.get("failed_or_unsupported", 0) == 0 and candidate.is_file()
                and (model is None or record.get("model") == model)
                and (language is None or record.get("language") == language)):
            return candidate
    return None


def check_ocr_report(md: Path):
    report_file = md.parent.parent / "visual_analysis.json"
    if not report_file.exists():
        raise RuntimeError(f"缺少视觉处理报告，不上传未验证的结果：{report_file}")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise RuntimeError(f"OCR/视觉阶段未完整成功（{report.get('status')}），停止入库：{report_file}")


def parser():
    p = argparse.ArgumentParser(
        description="本地 OCR-RAG：PDF → MinerU → 视觉增强 → Dify V2 自动入库")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--pdf", type=Path, help="指定一篇 PDF；省略时处理 data/input 下所有 PDF")
    group.add_argument("--from-zip", type=Path, help="复用已有 MinerU ZIP（跳过 MinerU PDF 解析）")
    group.add_argument("--md", type=Path, help="只将已有 enhanced_markdown.md 入库")
    p.add_argument("--prepare-only", action="store_true", help="生成增强/结构化结果，但不上传 Dify")
    p.add_argument("--force-ocr", action="store_true", help="忽略已完成的 OCR 缓存，重新解析 PDF")
    p.add_argument("--skip-smoke", action="store_true", help="不做 Dify 召回自测")
    p.add_argument("--mineru-exe", type=str, help="可选：指定已安装的 mineru-kit.exe 路径")
    p.add_argument("--dataset-id", type=str, help="可选：覆盖 DIFY_V2_ID")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--model", default="qwen3-vl:4b-instruct")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--language", choices=("zh", "en"), default="zh")
    p.add_argument("--vl-timeout", type=int, default=300)
    p.add_argument("--mineru-timeout", type=int, default=1800)
    p.add_argument("--fail-fast", action="store_true", help="单张图片识别失败立即终止")
    p.add_argument("--include-equations", action="store_true")
    return p


def execute(args):
    # All paths and existing V2 upload state live under the project root.
    # Works even when called from C:\Users\... instead of PyCharm Terminal.
    os.chdir(ROOT)
    pdfs: list[Path] = []
    if args.pdf:
        pdfs = [args.pdf]
    elif args.from_zip or args.md:
        pass
    else:
        pdfs = sorted(DEFAULT_INPUT.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"未发现 PDF：请把论文放到 {DEFAULT_INPUT}，然后运行 python main.py")
    sources = [("pdf", p) for p in pdfs]
    if args.from_zip:
        sources.append(("zip", args.from_zip))
    elif args.md:
        sources.append(("md", args.md))
    failures = []
    for kind, raw_path in sources:
        path = raw_path.resolve()
        print(f"\n==== 开始处理：{path.name} ====", flush=True)
        try:
            if not path.is_file():
                raise FileNotFoundError(f"文件不存在：{path}")
            if kind == "md":
                md = path
            else:
                md = (find_completed_pdf(path, args.output_root, model=args.model, language=args.language)
                      if kind == "pdf" and not args.force_ocr else None)
                if md is not None:
                    print(f"[OCR] 已完成且 PDF 未变化，复用：{md}")
                else:
                    print("[OCR] MinerU ZIP 缓存检查 + Ollama 视觉增强开始" if kind == "pdf"
                          else "[OCR] 解包 MinerU ZIP + Ollama 视觉增强开始")
                    md = run_ocr(
                        pdf=path if kind == "pdf" else None,
                        source_zip=path if kind == "zip" else None,
                        output_root=args.output_root,
                        mineru_exe=args.mineru_exe,
                        mineru_timeout=args.mineru_timeout,
                        ollama_url=args.ollama_url,
                        model=args.model,
                        language=args.language,
                        vl_timeout=args.vl_timeout,
                        fail_fast=args.fail_fast,
                        include_equations=args.include_equations,
                        force_ocr=args.force_ocr,
                    )
                check_ocr_report(md)
            result = ingest_markdown(md, prepare_only=args.prepare_only,
                                     dataset_id=args.dataset_id,
                                     skip_smoke=args.skip_smoke)
            print(f"✅ {path.name}: {result['status']}", flush=True)
        except (Exception, KeyboardInterrupt) as exc:
            failures.append((str(path), str(exc)))
            print(f"❌ {path.name} 失败：{exc}", file=sys.stderr, flush=True)
            if isinstance(exc, KeyboardInterrupt) or len(sources) == 1:
                break
    if failures:
        print("失败项目：" + ", ".join(Path(p).name for p, _ in failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(execute(parser().parse_args()))
