**Agentic Document Extraction**
- Extracts structured Purchase Order data, Markdown, and JSON page coordinates from PDFs
- Designed to be provider-agnostic via environment configuration; defaults target OpenAI + Agents SDK

**Features**
- Purchase Order extraction (schema enforced via Pydantic)
- Markdown rendering via text or vision
- JSON chunk coordinates via either vision or PyMuPDF (fitz) fallback
- Async, concurrent page/window processing
- Simple CLI for extraction and Q&A highlighting

**Install**
- Python 3.10+
- System dependency for `pdf2image`: install Poppler (e.g., `brew install poppler` on macOS, `apt-get install poppler-utils` on Debian/Ubuntu)
- Optional: Ghostscript can improve PDF parsing in some cases
- Then install Python deps:
- `pip install -r requirements.txt`

**Environment**
- Copy `.env.example` to `.env` and set values:
- `OPENAI_API_KEY` (if using OpenAI)
- `OPENAI_BASE_URL` (optional; proxies/self-hosted gateways)
- `LLM_MODEL`, `VISION_MODEL` (e.g., `gpt-5-mini`)
- `MAX_CONCURRENCY` (tune concurrency)

**CLI: Extraction**
- `python cli_extract.py <path/to.pdf> [--out ./output] [--po] [--markdown] [--json] [--vision] [--fitz] [--text-model ...] [--vision-model ...] [--max-concurrency N] [--po-window-pages W] [--po-window-stride S]`
- If no flags are provided, it defaults to `--po` only.
- Vision vs. fitz:
- `--vision` uses the model to return Markdown and (optionally) JSON chunks
- `--fitz` uses PyMuPDF to compute text chunks and coordinates (no vision)
- You can mix `--markdown` and `--json` with either path

**CLI: Q&A With Highlights**
- After extraction with `--json`, ask a question and render highlighted evidence:
- `python cli_qa.py <path/to.pdf> <path/to/output/<name>.json> "What is the PO total?" --out ./output/highlights --model gpt-5-mini`
- Produces annotated PNGs for cited pages and prints the answer

**Library Usage**
- Purchase Order extraction:
- `from agentic_document_extraction import process_po_file_async`
- `res = await process_po_file_async("/path/to.pdf", do_po=True, do_markdown=True, do_json_coords=True, use_vision=True)`
- Q&A highlighting:
- `from agentic_document_query import ask_and_annotate_with_highlighter_async`
- `res = await ask_and_annotate_with_highlighter_async(pdf_path, question, chunks=chunks)`

**Provider Layer**
- The repo uses a provider adapter with a default implementation for OpenAI Agents SDK.
- Configure provider via env: `PROVIDER=openai_agents` (default).
- Files:
- `providers/base.py` — async `Provider` interface
- `providers/openai_agents.py` — OpenAI Agents implementation
- `providers/__init__.py` — `get_provider()` factory
- To add a new provider:
- Implement a class extending `Provider` with `run_structured_text` and `run_structured_messages`
- Register it in `providers/__init__.py` (map your `PROVIDER` name)
- Ensure your client returns content parseable by the Pydantic output models

**OpenAI Setup (default provider)**
- Install the OpenAI Agents SDK that exposes `from agents import Agent, Runner`
- Set `OPENAI_API_KEY` in `.env`
- Choose models via `LLM_MODEL` and `VISION_MODEL`
- If using a proxy or Azure/OpenAI-compatible gateway, set `OPENAI_BASE_URL`

**Notes and Limits**
- Vision path via `pdf2image` requires Poppler installed
- Coordinates returned by the vision model are normalized [0,1] and validated
- The PyMuPDF (`fitz`) path returns approximated text-block coordinates without invoking a vision model

**Project Structure**
- `agentic_document_extraction.py` — core extraction (text/vision), PO schema, chunk generation
- `agentic_document_query.py` — chunk-driven QA + evidence highlighting
- `cli_extract.py` — CLI for extraction pipeline
- `cli_qa.py` — CLI for QA with page highlights
- `env.py` — .env loader/helpers
- `.env.example` — environment template
- `requirements.txt` — dependencies

**Troubleshooting**
- ImportError for `agents`:
- Install the OpenAI Agents SDK that exposes `from agents import Agent, Runner`.
- Some environments package it as `agents`, others as `openai-agents`; adjust as needed.
- Vision quality/speed:
- Increase `dpi` in `pdf2image` or `max_side`/`quality` in `_to_base64_jpeg`
- Missing text or tables in text path:
- Ensure `pdfplumber` is installed and Poppler is available
