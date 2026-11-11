"""Single-PDF orchestration; keeps original MinerU files and analysis artifacts."""
from __future__ import annotations

from datetime import datetime
import json
import shutil
import os
from zipfile import ZipFile, BadZipFile
from pathlib import Path

from .archive import unpack_zip
from .markdown import enhance_markdown, short_context
from .metadata import IMAGE_EXTS, SUPPORTED_VL_EXTS, inferred_metadata, lookup_image_metadata, normalize_reference
from .mineru import parse_pdf
from .utils import LOG, safe_stem, setup_logging, sha256_file, write_json
from .visual import OllamaVision, VISION_PARAMETERS


def already_parsed_tables(parsed_dir: Path, source_markdown: str) -> set[str]:
    """Images whose table cells were fully exported into the Markdown by MinerU.

    Only skip a screenshot when the matching structured table text is actually
    present in the final markdown. Incomplete/missing tables still use the VLM.
    """
    source = parsed_dir / "structured_content.json"
    if not source.is_file():
        return set()
    try:
        structured = json.loads(source.read_text(encoding="utf-8"))
        pages = structured.get("pages", [])
        normalized_md = " ".join(source_markdown.split())
        result: set[str] = set()
        for page in pages:
            for block in page.get("blocks", []):
                if block.get("type") != "table":
                    continue
                ref = block.get("image_source")
                text = block.get("content")
                if not isinstance(ref, str) or not isinstance(text, str):
                    continue
                # Ensure the table has a header + at least one data row.
                rows = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
                if len(rows) < 3:
                    continue
                if " ".join(text.split()) in normalized_md:
                    result.add(normalize_reference(ref))
        return result
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        LOG.warning("无法检查 MinerU 已解析表格：%s", exc)
        return set()


def valid_mineru_zip(path: Path) -> bool:
    """Verify the archive contains the essential MinerU Markdown before reusing."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with ZipFile(path) as z:
            names = z.namelist()
            markdowns = [n for n in names if n.split("/")[-1] == "markdown.md"]
            return len(markdowns) == 1 and bool(z.read(markdowns[0]).strip()) and z.testzip() is None
    except (OSError, BadZipFile, RuntimeError, KeyError):
        return False


def source_run_dir(source: Path, output_root: Path) -> Path:
    return output_root / f"{safe_stem(source.stem)[:56]}_{sha256_file(source)[:10]}"


def find_reusable_archive(source: Path, output_root: Path) -> Path | None:
    """Find valid previous MinerU ZIP for identical PDF, including old _rerun directories.

    The source PDF hash is encoded in the directory's stable prefix. The ZIP
    is reused for the same bytes; archives from unrelated input never match.
    """
    prefix = source_run_dir(source, output_root).name
    candidates = sorted(output_root.glob(prefix + "*/mineru_output.zip"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for archive in candidates:
        folder = archive.parent.name
        if folder != prefix and not folder.startswith(prefix + "_rerun_"):
            continue
        if valid_mineru_zip(archive):
            return archive
    return None


def run(pdf: Path | None, source_zip: Path | None, output_root: Path,
        mineru_exe: str | None, mineru_timeout: int, ollama_url: str,
        model: str, language: str, vl_timeout: int, fail_fast: bool,
        include_equations: bool, dry_run: bool = False, force_ocr: bool = False) -> Path:
    name = pdf.stem if pdf else source_zip.stem  # type: ignore[union-attr]
    source = pdf if pdf else source_zip
    assert source is not None
    if not source.is_file():
        raise FileNotFoundError(f"找不到输入文件：{source}")
    run_dir = source_run_dir(source, output_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "pipeline.log")
    archive_path = run_dir / "mineru_output.zip"

    report: dict = {
        "status": "running", "started_at": datetime.now().isoformat(timespec="seconds"),
        "pdf": str(pdf) if pdf else None, "source_zip": str(source_zip) if source_zip else None,
        "model": model, "language": language,
        "model_parameters": VISION_PARAMETERS,
        "images": [],
    }
    try:
        if source_zip:
            if not valid_mineru_zip(source_zip):
                raise ValueError(f"指定的 MinerU ZIP 无效：{source_zip}")
            if source_zip.resolve() != archive_path.resolve() and not valid_mineru_zip(archive_path):
                shutil.copy2(source_zip, archive_path)
            LOG.info("复用现有 MinerU ZIP：%s", source_zip)
        else:
            if pdf is None:
                raise ValueError("需要提供 --pdf 或 --from-zip")
            report["pdf_sha256"] = sha256_file(pdf)
            if force_ocr or not valid_mineru_zip(archive_path):
                prior = None if force_ocr else find_reusable_archive(pdf, output_root)
                if prior is not None and prior.resolve() != archive_path.resolve():
                    shutil.copy2(prior, archive_path)
                    LOG.info("[缓存] 复用过去成功解析的 MinerU ZIP：%s", prior)
                else:
                    LOG.info("[OCR] 无可复用解析结果，调用 MinerU")
                    # A failed attempt must never leave a partial ZIP that looks successful.
                    tmp_zip = run_dir / "mineru_output.pending.zip"
                    tmp_zip.unlink(missing_ok=True)
                    try:
                        parse_pdf(pdf, tmp_zip, run_dir / "mineru.log", mineru_exe, mineru_timeout)
                        if not valid_mineru_zip(tmp_zip):
                            raise RuntimeError("MinerU 返回了不可读取或不完整的 ZIP")
                        os.replace(tmp_zip, archive_path)
                    finally:
                        tmp_zip.unlink(missing_ok=True)
            else:
                LOG.info("[缓存] 本地 MinerU ZIP 有效，跳过 PDF 解析")
        parsed_dir = run_dir / "parsed"
        if not (parsed_dir / "markdown.md").is_file():
            parsed_dir = unpack_zip(archive_path, parsed_dir)
        else:
            LOG.info("[缓存] MinerU ZIP 已解压，跳过重复解压")
        md_path = parsed_dir / "markdown.md"
        source_markdown = md_path.read_text(encoding="utf-8")
        metadata = lookup_image_metadata(parsed_dir)
        LOG.info("结构化数据中找到 %d 个图片引用信息", len(metadata))
        parsed_tables = already_parsed_tables(parsed_dir, source_markdown)
        LOG.info("MinerU 已完整转换成 Markdown 的表格截图：%d 张（避免重复识别）", len(parsed_tables))

        images_dir = parsed_dir / "images"
        images = sorted(p for p in images_dir.rglob("*") if p.is_file() and
                        p.suffix.lower() in IMAGE_EXTS) if images_dir.exists() else []
        LOG.info("发现 %d 张图片", len(images))
        entries: dict[str, dict] = {}
        image_cache_path = run_dir / "vision_cache.json"
        try:
            cache = json.loads(image_cache_path.read_text(encoding="utf-8")) if image_cache_path.exists() else {}
        except (ValueError, OSError):
            cache = {}
        if (cache.get("model") != model or cache.get("language") != language or
                cache.get("vision_parameters") != VISION_PARAMETERS or
                cache.get("include_equations") != include_equations):
            cache = {"model": model, "language": language,
                     "vision_parameters": VISION_PARAMETERS,
                     "include_equations": include_equations, "images": {}}
        cached_images = cache.setdefault("images", {})
        client = OllamaVision(ollama_url, model, language, timeout=vl_timeout)
        try:
            for i, file in enumerate(images, start=1):
                ref = normalize_reference(file.relative_to(parsed_dir).as_posix())
                meta = inferred_metadata(ref, metadata)
                entry = {"path": ref, "kind": meta.get("kind", "image"),
                         "page": meta.get("page"), "page_metadata_source": meta.get("source", meta.get("page_source")),
                         "sha256": sha256_file(file)}
                entries[ref] = entry
                LOG.info("[%d/%d] %s | %s | 第 %s 页", i, len(images), ref,
                         entry["kind"], entry["page"] or "? ")
                if entry["kind"] == "table" and ref in parsed_tables:
                    entry["status"] = "already_parsed_table"
                    LOG.info("表格正文已在 Markdown 中，跳过重复视觉识别")
                    continue
                if entry["kind"] == "equation" and not include_equations:
                    entry["status"] = "skipped_equation"
                    LOG.info("公式截图：保留 MinerU 结果，不再自由描述")
                    continue
                if file.suffix.lower() not in SUPPORTED_VL_EXTS:
                    entry.update(status="unsupported_format", error=f"不支持的视觉输入格式：{file.suffix}")
                    continue
                if dry_run:
                    entry.update(status="dry_run", error="dry-run 未调用 Ollama")
                    continue
                cached = cached_images.get(ref)
                if cached and cached.get("sha256") == entry["sha256"] and cached.get("status") == "ok":
                    entries[ref] = dict(cached)
                    LOG.info("[缓存] 已完成图片识别，直接复用：%s", ref)
                    continue
                try:
                    result = client.analyze(file, entry["kind"], short_context(source_markdown, ref))
                    entry.update(status="ok", **result)
                    LOG.info("识别完成（%.2f 秒）", entry["latency_seconds"])
                    cached_images[ref] = dict(entry)
                    write_json(image_cache_path, cache)  # checkpoint after EACH successful image
                except Exception as exc:
                    entry.update(status="error", error=str(exc))
                    LOG.exception("识别失败：%s", ref)
                    if fail_fast:
                        raise
        finally:
            client.unload()

        enhanced, stats = enhance_markdown(source_markdown, entries)
        # The output file sits inside parsed/ so references like images/foo.jpg still resolve.
        output_md = parsed_dir / "enhanced_markdown.md"
        output_md.write_text(enhanced, encoding="utf-8")
        report["images"] = list(entries.values())
        report["statistics"] = {
            "image_count": len(images), "ok": sum(e["status"] == "ok" for e in entries.values()),
            "failed_or_unsupported": sum(e["status"] in ("error", "unsupported_format") for e in entries.values()),
            "skipped_equations": sum(e["status"] == "skipped_equation" for e in entries.values()),
            "already_parsed_tables": sum(e["status"] == "already_parsed_table" for e in entries.values()),
            **stats,
        }
        if dry_run:
            final_status = "dry_run"
        elif report["statistics"]["failed_or_unsupported"]:
            final_status = "partial_success" if report["statistics"]["ok"] else "failed_visual_analysis"
            if not report["statistics"]["ok"]:
                LOG.error("所有需识别图片均未成功；Markdown 仅保留 MinerU 的已识别正文，请勿当作视觉增强已完成")
        else:
            final_status = "complete"
        report.update(status=final_status, finished_at=datetime.now().isoformat(timespec="seconds"),
                      output_markdown=str(output_md.resolve()))
        write_json(run_dir / "visual_analysis.json", report)
        LOG.info("增强 Markdown：%s", output_md)
        LOG.info("完成：%s", json.dumps(report["statistics"], ensure_ascii=False))
        return output_md
    except Exception as exc:
        report.update(status="error", error=str(exc),
                      finished_at=datetime.now().isoformat(timespec="seconds"))
        write_json(run_dir / "failure_report.json", report)
        LOG.exception("处理失败；文件及日志保留在：%s", run_dir)
        raise
    finally:
        # Windows cannot delete/move log files while an active FileHandler holds them.
        for handler in list(LOG.handlers):
            LOG.removeHandler(handler)
            handler.close()
