import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from ocr_rag_pipeline.archive import ArchiveError, unpack_zip
from ocr_rag_pipeline.markdown import enhance_markdown, short_context
from ocr_rag_pipeline.metadata import inferred_metadata, lookup_image_metadata
from ocr_rag_pipeline.pipeline import run


class TestPipeline(unittest.TestCase):
    def test_zip_path_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            archive = Path(d) / 'bad.zip'
            with ZipFile(archive, 'w') as z:
                z.writestr('../escape.txt', 'attack')
                z.writestr('markdown.md', 'hello')
            with self.assertRaises(ArchiveError):
                unpack_zip(archive, Path(d) / 'out')

    def test_infer_zero_based_page_and_kind(self):
        info = inferred_metadata('images/page_3_chart_0.jpg', {})
        self.assertEqual(info['page'], 4)
        self.assertEqual(info['kind'], 'chart')
        self.assertEqual(inferred_metadata('images/unlabeled.jpg', {})['kind'], 'image')

    def test_page_metadata_priority(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            root.joinpath('structured_content.json').write_text(json.dumps({
                'pages': [{'page_idx': 1, 'blocks': [{
                    'type': 'chart', 'content': '![test](images/page_3_chart_0.jpg)'
                }]}]
            }), encoding='utf-8')
            matched = lookup_image_metadata(root)
            self.assertEqual(inferred_metadata('images/page_3_chart_0.jpg', matched)['page'], 2)

    def test_inline_and_unanchored_annotations(self):
        original = '# Paper\n\n![](images/page_3_chart_0.jpg)\n\nFigure 3: Original caption\n'
        entries = {
            'images/page_3_chart_0.jpg': {
                'path': 'images/page_3_chart_0.jpg', 'status': 'ok', 'kind': 'chart',
                'page': 4, 'description': 'Visual Input: 78.40%'
            },
            'images/other.jpg': {
                'path': 'images/other.jpg', 'status': 'ok', 'kind': 'image',
                'page': None, 'description': 'Unanchored figure'
            },
            'images/page_2_equation_1.jpg': {
                'path': 'images/page_2_equation_1.jpg', 'status': 'skipped_equation',
                'kind': 'equation', 'page': 3
            },
        }
        output, stats = enhance_markdown(original, entries)
        self.assertIn('Visual Input: 78.40%', output)
        self.assertIn('PDF 第 4 页', output)
        self.assertIn('Figure 3: Original caption', output)
        self.assertIn('Unanchored figure', output)
        self.assertEqual(stats['anchored_unique_images'], 1)
        self.assertEqual(stats['unanchored_images'], 1)
        self.assertNotIn('page_2_equation_1', output)
        self.assertIn('Figure 3', short_context(original, 'images/page_3_chart_0.jpg'))

    def test_from_zip_mock_vlm_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive = root / 'example.zip'
            with ZipFile(archive, 'w') as z:
                z.writestr('markdown.md', '# Test\n\n![](images/page_0_chart_0.jpg)\n\nFigure 1.\n')
                z.writestr('structured_content.json', json.dumps({
                    'pages': [{'page_idx': 0, 'blocks': [{'type':'chart',
                        'content': '![img](images/page_0_chart_0.jpg)'}]}]}))
                z.writestr('images/page_0_chart_0.jpg', b'fake-image')
            with patch('ocr_rag_pipeline.pipeline.OllamaVision') as mock_class:
                client = mock_class.return_value
                client.analyze.return_value = {
                    'description': 'Trend grows from 1 to 2.', 'latency_seconds': 1.0,
                    'prompt': 'test', 'model': 'qwen3-vl:4b'
                }
                output = run(pdf=None, source_zip=archive, output_root=root / 'output',
                             mineru_exe=None, mineru_timeout=120, ollama_url='http://localhost:11434',
                             model='qwen3-vl:4b', language='zh', vl_timeout=2,
                             fail_fast=False, include_equations=False)
                client.analyze.assert_called_once()
                client.unload.assert_called_once()
            self.assertIn('Trend grows from 1 to 2.', output.read_text(encoding='utf-8'))
            report = json.loads((output.parent.parent / 'visual_analysis.json').read_text(encoding='utf-8'))
            self.assertEqual(report['statistics']['ok'], 1)
            self.assertEqual(report['statistics']['unanchored_images'], 0)


if __name__ == '__main__':
    unittest.main()
