"""Grounded figure annotations for MinerU Markdown.

V6: Associate caption-specific author text with the same figure, without asking
small vision models to verify diagram topology. Quotes are exact substrings of
original Markdown; unsupported model geometry remains only in visual_analysis.json.
No new packages or model are required.
"""
from __future__ import annotations

import re
from pathlib import Path

from .metadata import normalize_reference

MD_IMAGE_RE = re.compile(
    r'!\[[^\]\n]*\]\(\s*(?:<(?P<angle>[^>]+)>|(?P<plain>[^)\s]+))'
    r'(?:\s+["\'][^)]*["\'])?\s*\)'
)
HTML_IMAGE_RE = re.compile(r'<img\b[^>]*?\bsrc\s*=\s*["\'](?P<src>[^"\']+)["\'][^>]*>', re.I)
CAPTION_AFTER_RE = re.compile(
    r'^\s*(?P<caption>(?:Figure|Fig\.?|图)\s*\d+[A-Za-z]?\s*[:：.]\s*[^\r\n]+)', re.I
)
CAPTION_BEFORE_RE = re.compile(
    r'(?P<caption>(?:Figure|Fig\.?|图)\s*\d+[A-Za-z]?\s*[:：.]\s*[^\r\n]+)\s*$', re.I
)
FIGURE_ID_RE = re.compile(r'^\s*(?:Figure|Fig\.?|图)\s*(\d+[A-Za-z]?)\b', re.I)
HEADING_RE = re.compile(r'(?m)^#{1,6}\s+')
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?。！？])\s+(?=[A-Z\u4e00-\u9fff])')
_WORD_RE = re.compile(r'[A-Za-z]{3,}|[\u4e00-\u9fff]{2,}')


def iter_image_markers(markdown: str):
    for match in MD_IMAGE_RE.finditer(markdown):
        yield match.start(), match.end(), normalize_reference(match.group('angle') or match.group('plain'))
    for match in HTML_IMAGE_RE.finditer(markdown):
        yield match.start(), match.end(), normalize_reference(match.group('src'))


def _caption_for_marker(markdown: str, start: int, end: int,
                        previous_end: int, next_start: int) -> tuple[str, int]:
    after = markdown[end:min(next_start, end + 1100)]
    match = CAPTION_AFTER_RE.match(after)
    if match:
        return match.group('caption').strip(), end + match.end('caption')
    before = markdown[max(previous_end, start - 1100):start]
    match = CAPTION_BEFORE_RE.search(before)
    if match:
        return match.group('caption').strip(), end
    return '', end


def figure_markers(markdown: str):
    markers = sorted(iter_image_markers(markdown))
    for i, (start, end, ref) in enumerate(markers):
        prev_end = markers[i - 1][1] if i else 0
        next_start = markers[i + 1][0] if i + 1 < len(markers) else len(markdown)
        caption, pos = _caption_for_marker(markdown, start, end, prev_end, next_start)
        yield start, end, ref, caption, pos


def short_context(md: str, ref: str) -> str:
    """Only this figure's caption goes into the VLM prompt, never nearby figures."""
    for _start, _end, path, caption, _pos in figure_markers(md):
        if path == normalize_reference(ref):
            return caption
    return ''


def _source_paragraphs(md: str):
    """Yield exact prose paragraphs. Exclude headings, captions, formulas and references."""
    # Bibliography references are not evidence for what a particular figure depicts.
    body = re.split(r'(?mi)^#{1,6}\s+(?:\d+\.?\s+)?REFERENCES\b', md, maxsplit=1)[0]
    for match in re.finditer(r'[^\r\n]+(?:\n(?!\s*\n)[^\r\n]+)*', body):
        para = match.group(0).strip()
        if not para or para.startswith(('!', '|', '#', '$$', '>', '<')):
            continue
        if CAPTION_AFTER_RE.match(para):
            continue
        # Only ordinary prose, not bare math or headings.
        if len(para) < 55 or para.count(' ') < 6:
            continue
        yield para


def _figure_ref_re(fig_id: str) -> re.Pattern:
    # No substring match: Fig. 1 must not match Fig. 11.
    return re.compile(r'(?:\bfig(?:ure)?\.?\s*|图\s*)' + re.escape(fig_id) + r'(?![0-9A-Za-z])', re.I)


def _select_source_sentences(para: str, ref_re: re.Pattern,
                             caption_terms: set[str], limit: int = 790) -> str:
    """Select supported sentences *verbatim*; do not paraphrase connections."""
    # There are often 5-8 sentences per scientific paragraph. Extract a small
    # number with reference/diagram-specific terms and preserve their order.
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(para) if s.strip()]
    if not sentences:
        return ''
    ranked = []
    for i, s in enumerate(sentences):
        terms = {w.casefold() for w in _WORD_RE.findall(s)}
        overlap = len(terms & caption_terms)
        scientific = bool(re.search(r'(?:\$|\\mathbf|\bhid(?:den)?\b|\blayer\b|\bnodes?\b|\bdimensions?\b|\bfeature\b|\bconnect|\bcombin|\bmodel\b|\bclassif|\baccurac)', s, re.I))
        score = 10 * bool(ref_re.search(s)) + min(overlap, 4) + 2 * scientific
        # Preserve the pivotal relation statement when a scientific paragraph
        # actually defines how two branches/layers are joined. This is source
        # prose, NOT a model-inferred edge.
        if re.search(r'\b(?:combined|merge[sd]?|additional hidden layer|two models|joint hidden layer|2-node|4,096-dimensional)\b', s, re.I):
            score += 7
        # A model's crucial technical details often follow the figure reference.
        if i and ref_re.search(sentences[i-1]):
            score += 3
        ranked.append((score, i, s))
    best = sorted(ranked, key=lambda x: (-x[0], x[1]))
    picked = []
    used = 0
    for score, i, s in best:
        if score < 2 or len(picked) >= 4:
            continue
        if used + len(s) > limit:
            continue
        picked.append((i, s))
        used += len(s)
    return ' '.join(s for _i, s in sorted(picked))


def source_evidence(md: str, caption: str, max_paragraphs: int = 2) -> list[str]:
    """Return matching author sentences from the Markdown, not generated claims.

    Require an explicit figure reference in each selected paragraph. Never use
    a generic nearest-section guess; unsupported figures simply get no quote.
    """
    fig_match = FIGURE_ID_RE.match(caption or '')
    if not fig_match:
        return []
    figure_id = fig_match.group(1)
    ref_re = _figure_ref_re(figure_id)
    terms = {w.casefold() for w in _WORD_RE.findall(caption)} - {
        'figure', 'fig', 'the', 'and', 'with', 'for', 'from', 'our', 'that', 'this', 'image',
    }
    candidates = []
    for para in _source_paragraphs(md):
        if not ref_re.search(para):
            continue
        # Explicit matching prose only. Reuse original string without rewrites.
        sel = _select_source_sentences(para, ref_re, terms)
        if not sel:
            continue
        words = {w.casefold() for w in _WORD_RE.findall(para)}
        # Technical descriptions outrank brief general mentions.
        domain = len(words & terms)
        detail = bool(re.search(r'\$|\\mathbf|\bhidden layer\b|\b4096\b|\b2-node\b', para, re.I))
        candidates.append((domain + 3 * detail, len(sel), sel))
    candidates.sort(key=lambda x: (-x[0], -x[1]))
    output = []
    for _score, _length, quoted in candidates:
        if quoted not in output:
            output.append(quoted)
        if len(output) >= max_paragraphs:
            break
    return output


def _normalize_math(text: str) -> str:
    """Conservative LaTeX normalization for audits, never for publication."""
    s = re.sub(r'\\(?:mathbf|mathrm|mathit)\s*\{\s*([A-Za-z])\s*\}', r'\1', text)
    s = re.sub(r'\s*_\s*\{\s*([A-Za-z0-9]+)\s*\}', r'_\1', s)
    s = re.sub(r'\s*\^\s*\{\s*\(\s*(\d+)\s*\)\s*\}', r'^\1', s)
    s = re.sub(r'\s*\^\s*\(\s*(\d+)\s*\)', r'^\1', s)
    s = re.sub(r'\s*\^\s*\{\s*(\d+)\s*\}', r'^\1', s)
    return s.replace(' ', '')


def symbol_audit(info: dict, excerpts: list[str]) -> dict:
    """A *non-authoritative* audit, only for JSON diagnostics.

    Missing mentions are NOT proof an image label is wrong, since prose may omit
    real diagram nodes. No invented relation becomes 'verified' here.
    """
    if info.get('extraction_task') != 'diagram':
        return {}
    facts = info.get('structured_facts') or {}
    if not isinstance(facts, dict) or not excerpts:
        return {}
    symbol_re = re.compile(r'(?<![A-Za-z0-9])(?:h|v)(?:_[A-Za-z0-9]+)?(?:\^\d+)?(?![A-Za-z0-9])')
    prose_symbols = set(symbol_re.findall(_normalize_math(' '.join(excerpts))))
    model_symbols = set()
    for item in facts.get('visible_nodes', []) + facts.get('ordered_modules', []):
        if isinstance(item, str):
            model_symbols.update(symbol_re.findall(_normalize_math(item)))
    unmatched = sorted(model_symbols - prose_symbols)
    return {
        'model_symbols_also_in_selected_prose': sorted(model_symbols & prose_symbols),
        'model_symbols_not_in_selected_prose': unmatched,
        'note': '原文未出现不等于图中不存在；未经人工或独立视觉核验，不证明模型连线正确。',
    }


def annotation(info: dict, caption: str = '', evidence: list[str] | None = None) -> str:
    if info.get('status') != 'ok':
        return ''
    body = (info.get('description') or '').strip()
    quotes = evidence or []
    if not body and not quotes:
        return ''
    page = info.get('page')
    page_label = f'PDF 第 {page} 页' if isinstance(page, int) else 'PDF 页码未确定'
    fig_id = FIGURE_ID_RE.match(caption) if caption else None
    figure_label = f'Figure {fig_id.group(1)}' if fig_id else Path(info['path']).name
    kind = info.get('kind', 'image')
    header = f'**{figure_label}｜{page_label}｜图像辅助信息（{kind}；{Path(info["path"]).name}）**'
    body = re.sub(r'^\s*(?:可见内容|图像内容|图像观察)\s*[:：]\s*', '', body)
    blocks = [header]
    if body:
        blocks.append('**视觉模型观察（未核验）：**' + body)
    if quotes:
        # Quotation is verbatim from ORIGINAL Markdown, not an LLM rewrite.
        blocks.append('**论文正文关联证据（原文摘录，不代表图像连线已核验）：**\n' +
                      '\n'.join('> ' + q.replace('\n', ' ') for q in quotes))
    return '\n\n' + '\n\n'.join(blocks) + '\n'


def enhance_markdown(original: str, info_by_path: dict[str, dict]) -> tuple[str, dict]:
    replacements = []
    inserted: set[str] = set()
    markers = list(figure_markers(original))
    sourced = 0
    for _start, _end, path, caption, position in markers:
        info = info_by_path.get(path)
        if info is None or info.get('status') != 'ok':
            continue
        # Append text evidence only for architecture/flow diagrams. Other
        # figures already have model observations and the original caption.
        task = info.get('extraction_task', '')
        is_diagram = task == 'diagram'
        quotes = source_evidence(original, caption) if is_diagram else []
        if quotes:
            sourced += 1
            info['source_grounding'] = {
                'figure': FIGURE_ID_RE.match(caption).group(1),
                'quoted_original_sentences': quotes,
                'symbol_audit': symbol_audit(info, quotes),
                'status': 'source_text_attached_not_independent_vision_verified',
            }
        addition = annotation(info, caption, quotes)
        if not addition:
            continue
        replacements.append((position, addition))
        inserted.add(path)

    enhanced = original
    for position, addition in sorted(replacements, reverse=True):
        enhanced = enhanced[:position] + addition + enhanced[position:]
    unanchored = [info for path, info in info_by_path.items()
                  if path not in inserted and info.get('status') == 'ok']
    if unanchored:
        enhanced += '\n\n---\n\n## 未与正文图像对应的辅助解析（位置未确定）\n\n'
        for info in sorted(unanchored, key=lambda i: (i.get('page') is None,
                                                       i.get('page') or 0, i['path'])):
            enhanced += annotation(info)
    return enhanced, {
        'markers': len(markers),
        'anchored_unique_images': len(inserted),
        'unanchored_images': len(unanchored),
        'unanchored_paths': [i['path'] for i in unanchored],
        'diagrams_with_original_text_evidence': sourced,
    }
