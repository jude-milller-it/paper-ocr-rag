"""Ollama Qwen3-VL visual evidence extraction (no extra dependencies).

Structured image observations are useful for retrieval, but scientific network
symbols and inferred links are not reliable enough to present as verified facts.
The original model extraction is retained in JSON for audit; uncertain network
links do not enter searchable Markdown. A single vision pass is used by default.

Replace only ocr_rag_pipeline/visual.py in the existing project.
"""
from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .utils import LOG

VISION_PARAMETERS = {
    "api_endpoint": "/api/chat",
    "format": "per-kind JSON schema",
    "think": False,
    "stream": False,
    "keep_alive": "3m",
    "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 1536},
    "diagram_independent_passes": 0,
    "diagram_edges_published": False,
    "retry_on_inconsistency": True,
}


class OllamaError(RuntimeError):
    pass


# Simple JSON schemas are more manageable for the 4B model than a large
# catch-all schema. Output is parsed/validated again by local Python code.
def _str():
    return {"type": "string"}


def _str_array():
    return {"type": "array", "items": _str()}


def _schema(properties: dict) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


DIAGRAM_SCHEMA = _schema({
    "visible_nodes": _str_array(),
    "ordered_modules": _str_array(),
    "visible_edges": {"type": "array", "items": _schema({"from": _str(), "to": _str()})},
    "printed_dimensions": _str_array(),
    "uncertain": _str_array(),
})

PANEL_SCHEMA = _schema({
    "rows": {"type": "integer", "minimum": 0, "maximum": 30},
    "columns": {"type": "integer", "minimum": 0, "maximum": 30},
    "image_count": {"type": "integer", "minimum": 0, "maximum": 500},
    "row_labels": _str_array(),
    "column_labels": _str_array(),
    "visible_objects": _str_array(),
    "uncertain": _str_array(),
})

CHART_SCHEMA = _schema({
    "chart_type": _str(),
    "x_axis": _str(),
    "y_axis": _str(),
    "data_points": {"type": "array", "items": _schema({"label": _str(), "value": _str(), "series": _str()})},
    "visible_trend": _str(),
    "uncertain": _str_array(),
})

TABLE_SCHEMA = _schema({
    "headers": _str_array(), "rows": _str_array(), "uncertain": _str_array(),
})

GENERAL_SCHEMA = _schema({
    "visual_kind": {"type": "string", "enum": ["diagram", "panel", "photo", "other"]},
    "visible_objects": _str_array(),
    "printed_labels": _str_array(),
    "visible_layout": _str(),
    "uncertain": _str_array(),
})

SCHEMAS = {"diagram": DIAGRAM_SCHEMA, "panel": PANEL_SCHEMA,
           "chart": CHART_SCHEMA, "table": TABLE_SCHEMA, "general": GENERAL_SCHEMA}

SYSTEM = ("你是科研图片的事实提取器，只依据当前图片实际可见的像素提取信息。"
          "附带图注只帮助判断当前图片类型，不能把图注内容冒充为亲眼读到的标签。"
          "不能参考任何其他图片或论文段落，不编造数字、层名、边、照片数量。"
          "只输出符合指定 JSON Schema 的 JSON；无法辨认的字段用空字符串、空数组或0。")

PROMPTS = {
    "diagram": (
        "类型：网络架构图/流程图。请先逐字提取图中可辨的节点/模块名称 visible_nodes，"
        "再提取明确写出的顺序 ordered_modules；printed_dimensions 仅列图中清晰印出的尺寸、数字及其节点。"
        "visible_edges 只填写可以逐条目视追踪箭头方向、并且两端节点名称可见的连接。"
        "箭头模糊或只是根据常识推断的连接必须省略，禁止给所有节点补完全连接。"
        "注意区分 h^(3) 与 h_m^(3)、上下标、fc14/fc15。"
        "若看不清一个值，放入 uncertain，不能用论文常识补出。"
    ),
    "panel": (
        "类型：多幅照片/子图拼图。先数实际可见的行 rows、列 columns、图片总张数 image_count。"
        "不要把标题、图注和文字标签算成一张照片。"
        "逐字提取图内实际印出的行列标签；visible_objects 按阅读顺序简短列出主要可见内容，"
        "不猜测分类是否正确、研究结果或图像拍摄目的。"
        "重要：自检 image_count 是否与 rows×columns 一致；如果有不规则排版、空格，"
        "计数必须基于真正看见的图片区块，无法确定写 0。"
    ),
    "chart": (
        "类型：科学统计图表。明确抄出图表类型和横纵轴原文（含单位）。"
        "data_points 每个点包括横轴标签 label、图上直接标出的值 value（保持原来的小数位）和系列名 series。"
        "看不清精确数据不要靠曲线位置猜小数，value 填空字符串。"
        "visible_trend 只能写肉眼可验证的升降趋势，不总结论文价值。"
        "注意 y 轴单位可能是 %，不要把 78.40 写成 0.784。"
    ),
    "table": (
        "类型：尚未转换成文字的表格。仅抄出看清的列头 headers 和数据行 rows（每行一项原文），"
        "不计算推断，不补造模糊单元格。"
    ),
    "general": (
        "提取当前图中可见的对象、版面和文字标签。"
        "若可见内容是流程图或多图拼贴，visual_kind 填 diagram 或 panel；"
        "否则 photo 或 other。visible_layout 只写可以直接看到的位置和排列。"
        "绝不凭图片推断论文实验结论。"
    ),
}


def choose_task(kind: str, caption: str) -> str:
    if kind in ("chart", "table"):
        return kind
    if kind == "equation":
        return "general"
    c = caption.lower()
    if any(key in c for key in ("architecture", "framework", "network structure",
                                "model structure", "workflow", "flow chart", "flowchart",
                                "approach overview", "three modules", "pipeline", "架构", "流程图", "模块结构")):
        return "diagram"
    if any(key in c for key in ("test images", "representative items", "examples", "sample images",
                                "representative images", "photo panels", "photos", "first row",
                                "图像样例", "多张照片", "示例图片", "图片样例")):
        return "panel"
    return "general"


def make_prompt(language: str, kind: str, context: str) -> str:
    """Compatible with prior callers; here 'kind' is an extraction task."""
    caption = (context or "无对应图注")[:500]
    if language == "zh":
        return f"原论文图注（独立原始文本）：{caption}\n当前图像任务：{PROMPTS[kind]}"
    # Keep the fields stable for English output, avoid translating exact labels.
    return ("This is one figure; its caption is context only: " + caption + "\n" +
            "Read exact *visible* labels and values. Never invent edges, counts or data. " +
            PROMPTS[kind] + " Respond in JSON; descriptions may be English.")


def _norm(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _distinct_strings(raw: object, limit: int = 40) -> list[str]:
    if not isinstance(raw, list):
        raise ValueError("预期字段是 JSON 数组")
    seen = set()
    result = []
    for item in raw[:limit]:
        if not isinstance(item, str):
            raise ValueError("列表成员必须是字符串")
        item = item.strip().replace("\n", " ")[:240]
        if item and _norm(item) not in seen:
            seen.add(_norm(item))
            result.append(item)
    return result


def parse_facts(content: str, task: str) -> tuple[dict, list[str]]:
    """Local schema and semantic validation, no external packages."""
    try:
        obj = json.loads(content)
    except ValueError as exc:
        raise ValueError(f"无效 JSON：{str(exc)[:160]}") from exc
    if not isinstance(obj, dict):
        raise ValueError("顶层必须是 JSON 对象")
    for key in SCHEMAS[task]["required"]:
        if key not in obj:
            raise ValueError(f"缺少字段 {key}")
    flags: list[str] = []
    if task == "panel":
        for k in ("rows", "columns", "image_count"):
            if type(obj[k]) is not int or not 0 <= obj[k] <= 500:
                raise ValueError(f"{k} 必须是非负整数")
        for k in ("row_labels", "column_labels", "visible_objects", "uncertain"):
            obj[k] = _distinct_strings(obj[k], 40)
        n = obj["rows"] * obj["columns"]
        if n and obj["image_count"] and n != obj["image_count"]:
            flags.append(f"数量矛盾：{obj['rows']}行×{obj['columns']}列={n}，但总数写{obj['image_count']}")
        if obj["rows"] and obj["row_labels"] and len(obj["row_labels"]) > obj["rows"]:
            flags.append("行标签数量多于行数")
        if obj["columns"] and obj["column_labels"] and len(obj["column_labels"]) > obj["columns"]:
            flags.append("列标签数量多于列数")
    elif task == "diagram":
        for k in ("visible_nodes", "ordered_modules", "printed_dimensions", "uncertain"):
            obj[k] = _distinct_strings(obj[k])
        edges = obj["visible_edges"]
        if not isinstance(edges, list):
            raise ValueError("visible_edges 必须是数组")
        nodes = {_norm(t) for t in obj["visible_nodes"] + obj["ordered_modules"]}
        valid_edges = []
        for edge in edges[:80]:
            if not isinstance(edge, dict) or not isinstance(edge.get('from'), str) or not isinstance(edge.get('to'), str):
                raise ValueError("visible_edges 中节点必须有 from/to 字符串")
            src, dst = edge['from'].strip(), edge['to'].strip()
            if not src or not dst or _norm(src) == _norm(dst):
                continue
            if _norm(src) not in nodes or _norm(dst) not in nodes:
                flags.append(f"连接端点不在已识别节点中：{src}→{dst}")
                continue
            valid_edges.append({"from": src, "to": dst})
        obj["visible_edges"] = valid_edges
    elif task == "chart":
        for key in ("chart_type", "x_axis", "y_axis", "visible_trend"):
            if not isinstance(obj[key], str):
                raise ValueError(f"{key} 必须是字符串")
            obj[key] = obj[key].strip()[:300]
        obj["uncertain"] = _distinct_strings(obj["uncertain"])
        points = obj["data_points"]
        if not isinstance(points, list):
            raise ValueError("data_points 必须是数组")
        clean_points = []
        for point in points[:100]:
            if not isinstance(point, dict) or not all(isinstance(point.get(k), str) for k in ("label", "value", "series")):
                raise ValueError("data_points 必须包含 label/value/series 字符串")
            label, value, series = (point[k].strip()[:130] for k in ("label", "value", "series"))
            # Accept values like 78.40, 78.4%, -0.4, 1,234.5 and confidence ranges.
            # Reject detailed guessed sentences as exact numeric points.
            if value and not re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?\s*%?", value):
                flags.append(f"非数值型读数，已省略：{label}={value[:35]}")
                value = ""
            if label or value:
                clean_points.append({"label": label, "value": value, "series": series})
        obj["data_points"] = clean_points
    elif task == "table":
        for key in ("headers", "rows", "uncertain"):
            obj[key] = _distinct_strings(obj[key], 80)
    else:
        if obj['visual_kind'] not in ("diagram", "panel", "photo", "other"):
            raise ValueError("visual_kind 非法")
        for key in ("visible_objects", "printed_labels", "uncertain"):
            obj[key] = _distinct_strings(obj[key])
        if not isinstance(obj['visible_layout'], str):
            raise ValueError("visible_layout 不是字符串")
        obj['visible_layout'] = obj['visible_layout'].strip()[:450]
    return obj, flags


def _clip(text: str, max_length: int = 800) -> str:
    return (text or "").strip()[:max_length]


# Notation may be produced by the model in TeX, plain ASCII or Unicode form.
# Treat each of these as an unverified mathematical/schematic label rather than
# a safely identified connection: h_m^(3), h_{m}^{(3)}, h^(3), hₘ³, v_m, h2.
_SYMBOLIC_NODE_RE = re.compile(
    r"(?<![A-Za-z0-9])[hv](?:"
    r"\s*_\s*(?:\{[^{}]{1,15}\}|[A-Za-z0-9]{1,6})|"
    r"\s*\^\s*(?:\{[^{}]{1,15}\}|\([^()]{1,15}\)|\d{1,3})|"
    r"[₀-₉ₐ-ₜ⁰¹²³⁴⁵⁶⁷⁸⁹]|"
    r"\d{1,3}"
    r")+(?![A-Za-z0-9])", re.IGNORECASE
)


def _is_symbolic_label(value: str) -> bool:
    return bool(_SYMBOLIC_NODE_RE.search(value))


def _is_complex_network(data: dict) -> bool:
    # Examine ALL extracted fields: the model may omit symbolic nodes from
    # visible_nodes while still putting invented symbols in ordered_modules.
    values = (list(data.get("visible_nodes", []))
              + list(data.get("ordered_modules", []))
              + list(data.get("printed_dimensions", [])))
    for edge in data.get("visible_edges", []):
        if isinstance(edge, dict):
            values.extend([edge.get("from", ""), edge.get("to", "")])
    return any(isinstance(x, str) and (
        _is_symbolic_label(x) or re.search(r"\b(?:multimodal\s+)?DBM\b", x, re.I)
    ) for x in values)


def render_facts(task: str, data: dict, quality_flags: list[str] | None = None) -> str:
    """Deterministic rendering prevents inconsistent free-form summaries."""
    parts = []
    if task == 'panel':
        rows, cols, total = data['rows'], data['columns'], data['image_count']
        contradiction = bool(quality_flags)
        if not contradiction:
            if rows and cols:
                parts.append(f"图像排列：{rows}行×{cols}列" + (f"，共{total}张" if total else ""))
            elif total:
                parts.append(f"可见子图约{total}张（行列无法确定）")
        if data['row_labels']:
            parts.append("图中行标签：" + "、".join(data['row_labels']))
        if data['column_labels']:
            parts.append("图中列标签：" + "、".join(data['column_labels']))
        if data['visible_objects']:
            parts.append("可见内容：" + "；".join(data['visible_objects'][:18]))
    elif task == 'diagram':
        nodes = data['visible_nodes']
        if _is_complex_network(data):
            # A 4B model can repeatedly hallucinate the SAME symbol (e.g.
            # h_m^(3) in a diagram that actually labels h^(3)). Do not trust
            # a second identical generation to establish correctness.
            safe_nodes = [n for n in nodes if not _is_symbolic_label(n)
                          and not re.fullmatch(r"[\d\W]+", n)
                          and '...' not in n and '…' not in n
                          and len(n) <= 75]
            if safe_nodes:
                parts.append("可辨认的普通文字标签（模型提取，非连线核验）：" +
                             "、".join(safe_nodes[:24]))
            # Do not put guessed symbols, raw numerical layer indices, module
            # ordering or network edges into searchable Markdown.
        else:
            if nodes:
                parts.append("图中标注的节点或模块（模型提取）：" + "、".join(nodes[:40]))
            if data['ordered_modules']:
                # A sequence of observed labels is NOT evidence that arrows
                # connect adjacent entries (e.g. conv1 ... conv13).
                parts.append("模型识别的模块排列（连线未核验）：" +
                             "、".join(data['ordered_modules'][:22]))
            if data['printed_dimensions']:
                parts.append("可见数值或维度（所属节点需核验）：" +
                             "；".join(data['printed_dimensions'][:30]))
        # Never publish raw inferred arrows as verified topology, regardless
        # of whether multiple same-model attempts happen to agree.
    elif task == 'chart':
        if data['chart_type']:
            parts.append("图表类型：" + data['chart_type'])
        if data['x_axis']:
            parts.append("横轴：" + data['x_axis'])
        if data['y_axis']:
            parts.append("纵轴：" + data['y_axis'])
        points = [p for p in data['data_points'] if p['value']]
        if points:
            fmt = []
            for p in points:
                lead = (p['series'] + " / " if p['series'] else "") + (p['label'] or "未标注点")
                fmt.append(lead + "=" + p['value'])
            parts.append("图中清晰读数：" + "；".join(fmt))
        if data['visible_trend']:
            parts.append("可见趋势：" + data['visible_trend'])
    elif task == 'table':
        if data['headers']:
            parts.append("列名：" + "、".join(data['headers']))
        if data['rows']:
            parts.append("可见数据行：" + "；".join(data['rows']))
    else:
        if data['visible_objects']:
            parts.append("图像可见对象：" + "、".join(data['visible_objects']))
        if data['printed_labels']:
            parts.append("图中文字标签：" + "、".join(data['printed_labels']))
        if data['visible_layout']:
            parts.append("可见布局：" + data['visible_layout'])
    return "。".join(_clip(t, 1700) for t in parts) + ("。" if parts else "")


class OllamaVision:
    def __init__(self, url: str, model: str, language: str = "zh",
                 timeout: int = 300, attempts: int = 2, num_ctx: int = 4096):
        self.url = url.rstrip("/")
        self.model = model
        self.language = language
        self.timeout = timeout
        self.attempts = max(1, attempts)
        self.num_ctx = num_ctx
        self._used = False

    def _post(self, payload: dict, timeout: int | None = None,
              endpoint: str = "/api/chat") -> dict:
        request = Request(self.url + endpoint,
                          data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=timeout or self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")[:800]
            raise OllamaError(f"Ollama HTTP {exc.code}: {details}") from exc
        except (URLError, TimeoutError, ConnectionError, OSError) as exc:
            raise OllamaError(f"Ollama 无法连接或超时：{self.url}（{exc}）") from exc
        except ValueError as exc:
            raise OllamaError("Ollama 返回了无效 JSON") from exc
        if 'error' in result:
            raise OllamaError(f"Ollama 错误：{result['error']}")
        return result

    def _read(self, image_b64: str, prompt: str, task: str) -> tuple[dict, list[str], dict]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt, "images": [image_b64]}],
            "format": SCHEMAS[task], "think": False, "stream": False,
            "keep_alive": "3m",
            "options": {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": 1536},
        }
        response = self._post(payload)
        message = response.get("message") or {}
        if not isinstance(message, dict):
            raise OllamaError("Ollama 返回异常 message 对象")
        content = message.get("content") or ""
        if not content.strip():
            raise OllamaError("模型空响应："
                              f"done_reason={response.get('done_reason')}, "
                              f"eval_count={response.get('eval_count')}, "
                              f"thinking_chars={len(str(message.get('thinking') or ''))}")
        try:
            data, flags = parse_facts(content, task)
        except ValueError as exc:
            raise OllamaError(f"模型结构化输出无效：{exc}") from exc
        info = {"eval_count": response.get("eval_count"),
                "prompt_eval_count": response.get("prompt_eval_count"),
                "done_reason": response.get("done_reason")}
        return data, flags, info

    def analyze(self, file: Path, kind: str, context: str = "") -> dict:
        if not file.is_file():
            raise FileNotFoundError(f"图像不存在：{file}")
        task = choose_task(kind, context)
        prompt = make_prompt(self.language, task, context)
        image_b64 = base64.b64encode(file.read_bytes()).decode("ascii")
        started = time.perf_counter()
        self._used = True
        flags_history: list[str] = []
        extraction = None
        stats = {}
        actual_attempts = 0
        for attempt in range(1, self.attempts + 1):
            actual_attempts = attempt
            try:
                # Each attempt is independent, with targeted correction of the
                # instruction only (do not send a contradictory prior answer).
                correction = "\n请先在脑中检查数量、数组和图片是否一致；不确定的填写0。" if attempt > 1 else ""
                data, flags, stats = self._read(image_b64, prompt + correction, task)
                if flags:
                    flags_history.extend(flags)
                    if attempt < self.attempts:
                        LOG.warning("结构化提取自相矛盾，正在重新识别：%s", "; ".join(flags))
                        continue
                extraction = data
                break
            except OllamaError as exc:
                flags_history.append(str(exc))
                if attempt == self.attempts:
                    raise
                LOG.warning("图像请求失败（%d/%d）：%s；重新请求", attempt, self.attempts, exc)
                time.sleep(1)
        if extraction is None:
            raise OllamaError("提取失败，未返回合法 JSON")

        # Captions can be absent or vague in arbitrary PDF files. If an initial
        # generic read identifies a diagram or panel, re-extract with that
        # task's narrower schema instead of indexing a shallow generic summary.
        if task == "general" and extraction["visual_kind"] in ("diagram", "panel"):
            discovered_task = extraction["visual_kind"]
            try:
                discovered, discovered_flags, discovered_stats = self._read(
                    image_b64, make_prompt(self.language, discovered_task, context),
                    discovered_task)
                actual_attempts += 1
                task = discovered_task
                extraction = discovered
                stats = discovered_stats
                flags_history.extend(discovered_flags)
            except OllamaError as exc:
                flags_history.append(f"分类后细节提取失败，使用保守通用描述：{exc}")

        # Do not make a second same-model pass for architecture edges: it cannot
        # independently verify topology, and unverified edges are not indexed.
        if task == "diagram":
            flags_history.append("架构连线未经过独立验证，因此未写入 Markdown")
            if _is_complex_network(extraction):
                flags_history.append("含复杂网络符号：数学节点及层级关系仅保留在原始 JSON，不能作为已核实事实")
            # Explicitly mark the JSON evidence as unverified as well. External
            # consumers must not silently treat extracted arrows as ground truth.
            extraction["topology_verified"] = False

        # Contradictory grid counts must never appear in indexed Markdown,
        # even if the corrective attempt also disagreed.
        visible_flags = []
        if task == 'panel':
            _, visible_flags = parse_facts(json.dumps(extraction, ensure_ascii=False), task)
        description = render_facts(task, extraction, visible_flags)
        if not description:
            raise OllamaError("JSON 有效，但没有可供安全入库的可见内容")
        if visible_flags:
            flags_history.extend(visible_flags)
        return {
            "description": description,
            "structured_facts": extraction,
            "quality_flags": list(dict.fromkeys(flags_history)),
            "extraction_task": task,
            "attempt_count": actual_attempts,
            "latency_seconds": round(time.perf_counter() - started, 2),
            "prompt": prompt,
            "model": self.model,
            **stats,
        }

    def unload(self) -> None:
        if not self._used:
            return
        try:
            response = self._post({"model": self.model, "prompt": "", "stream": False,
                                   "keep_alive": 0}, timeout=30, endpoint="/api/generate")
            LOG.info("视觉模型卸载请求已完成：%s（%s）", self.model,
                     response.get("done_reason", "done"))
        except OllamaError as exc:
            LOG.warning("自动卸载失败：%s；可手动执行 ollama stop %s", exc, self.model)
