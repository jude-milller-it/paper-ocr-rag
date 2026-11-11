# Local OCR and RAG for Research Papers

[中文说明](README_CN.md)

This pipeline turns research PDFs into a local, searchable knowledge base. MinerU extracts text, formulas, tables, and images. Qwen3-VL, served by Ollama, adds descriptions of figures and plots. Python structures the resulting evidence and syncs it to Dify for retrieval and question answering.

## Evidence organization

A single chunking strategy can separate a figure from its caption or mix several tables into one result. This project creates separate overview, figure, table, and section documents. A Dify workflow routes questions by evidence type, so a figure question can search the figure documents directly. Figure retrieval uses textual descriptions and captions.

On *Joint Image and Text Representation for Aesthetics*, the pipeline produced **22 knowledge documents**: one overview, five figures, two tables, and fourteen section documents. This is a single-paper integration check. Exact source citation and reliability across larger collections remain open.

## Run locally

1. Install MinerU with the `mineru-kit` CLI, Ollama, and Dify. Install Python's HTTP dependency with `python -m pip install requests`.
2. Make `qwen3-vl:4b-instruct`, `qwen3-embedding:0.6b`, and `qwen2.5:3b` available to Ollama. In Dify, create a knowledge base with the embedding model and set `DIFY_API_KEY` and `DIFY_V2_ID` for your instance. Set `DIFY_API_BASE` if your endpoint differs from `http://localhost:8080/v1`.
3. Import [the Dify workflow](OCR_RAG_QA_V2_Dify_1.17.yml), bind your knowledge base in all five retrieval nodes, and configure the answer model.
4. Ingest a paper with `python -X utf8 main.py --pdf path/to/paper.pdf`. Add `--prepare-only` to generate the documents locally before syncing to Dify. Once indexing completes, ask questions in the imported workflow.
