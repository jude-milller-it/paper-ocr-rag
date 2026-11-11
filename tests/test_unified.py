"""Tests for the single entrypoint, no live MinerU/Ollama/Dify required."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from main import check_ocr_report, execute, find_completed_pdf
from ocr_rag_pipeline.knowledge import make_records


class TestOneCommand(unittest.TestCase):
    def test_figure_and_table_atomized(self):
        raw = (
            "# Test Paper\n\n## ABSTRACT\n\nWe propose multimodal learning.\n\n"
            "![](images/page_1_image_0.jpg)\n\n"
            "Figure 1: Overview.\n\n"
            "**Figure 1｜PDF 第 2 页｜图像辅助信息（image；x）**\n\n"
            "**视觉模型观察（未核验）：** Joint representation.\n\n"
            "Table 1: Performance\n\n"
            "| Methods | Accuracy(%) |\n|---|---|\n| A | 78.19 |\n\n"
            "## 4. CONCLUSIONS\n\nModel is effective."
        )
        # CRLF from exported archives must also work.
        records = make_records(raw.replace("\n", "\r\n"), "paper_12345678")
        self.assertEqual(sum(x.kind == 'figure' for x in records), 1)
        self.assertEqual(sum(x.kind == 'table' for x in records), 1)
        self.assertEqual(sum(x.kind == 'overview' for x in records), 1)
        self.assertIn('78.19', next(x.content for x in records if x.kind == 'table'))

    def test_pdf_reuse_completed_ocr(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pdf = root / 'test.pdf'
            pdf.write_bytes(b'%PDF-1.4\ntest')
            import hashlib
            h = hashlib.sha256(pdf.read_bytes()).hexdigest()
            output = root / 'out' / 'batch'
            (output / 'parsed').mkdir(parents=True)
            md = output / 'parsed' / 'enhanced_markdown.md'
            md.write_text('# dummy', encoding='utf-8')
            (output / 'visual_analysis.json').write_text(json.dumps({
                'pdf_sha256': h, 'status': 'complete',
                'statistics': {'failed_or_unsupported': 0},
            }), encoding='utf-8')
            self.assertEqual(find_completed_pdf(pdf, root / 'out'), md)
            pdf.write_bytes(b'%PDF-1.4\nnew-content')
            self.assertIsNone(find_completed_pdf(pdf, root / 'out'))

    def test_failure_stops_before_dify(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            md = root / 'out' / 'parsed' / 'enhanced_markdown.md'
            md.parent.mkdir(parents=True)
            md.write_text('markdown', encoding='utf-8')
            (root / 'out' / 'visual_analysis.json').write_text(json.dumps({
                'status': 'partial_success'
            }), encoding='utf-8')
            with self.assertRaises(RuntimeError):
                check_ocr_report(md)

    def test_single_pdf_pipeline_calls_ingest_without_searching_latest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pdf = root / 'research.pdf'
            pdf.write_bytes(b'%PDF-1.4')
            md = root / 'run' / 'parsed' / 'enhanced_markdown.md'
            md.parent.mkdir(parents=True)
            md.write_text('# Test', encoding='utf-8')
            (root / 'run' / 'visual_analysis.json').write_text(json.dumps({
                'status': 'complete'
            }), encoding='utf-8')
            args = SimpleNamespace(pdf=pdf, from_zip=None, md=None,
                output_root=root / 'runs', force_ocr=False, mineru_exe=None,
                mineru_timeout=120, ollama_url='http://localhost:11434',
                model='qwen3-vl:4b-instruct', language='zh', vl_timeout=1,
                fail_fast=False, include_equations=False, prepare_only=False,
                dataset_id=None, skip_smoke=True)
            with patch('main.run_ocr', return_value=md) as ocr, \
                 patch('main.ingest_markdown', return_value={'status': 'indexed'}) as ingest:
                self.assertEqual(execute(args), 0)
                ocr.assert_called_once()
                ingest.assert_called_once()
                self.assertEqual(ingest.call_args.args[0], md)

if __name__ == '__main__':
    unittest.main()
