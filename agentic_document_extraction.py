from __future__ import annotations
import os, io, json, asyncio, hashlib, re, base64, fitz, time
from typing import Any, Dict, List, Optional, Tuple, Literal, Annotated, Union
from concurrent.futures import ThreadPoolExecutor

import pdfplumber
from pdf2image import convert_from_bytes
from PIL import Image

from pydantic import BaseModel, Field
from agents import Agent, Runner, function_tool, RunContextWrapper 
from openai.types.responses import ResponseInputImageParam, ResponseInputTextParam
from openai.types.responses.response_input_item_param import Message

# =============== Models (PO) ===============
class LineItem(BaseModel):
    line_no: Optional[str] = None
    part: str
    revision: Optional[str] = None
    description: Optional[str] = None
    qty: Optional[float] = None
    unit_price: Optional[float] = None
    extended_price: Optional[float] = None
    dpas_rating: Optional[str] = None
    delivery_date: Optional[str] = None
    anchor_id: Optional[str] = None

class PurchaseOrder(BaseModel):
    # page is used by the per-page extractor; dropped in final merge
    page: Optional[int] = None
    po_number: Optional[str] = None
    buyer: Optional[str] = None
    supplier_name: Optional[str] = None
    supplier_address: Optional[str] = None
    create_date: Optional[str] = None
    payment_terms: Optional[str] = None
    incoterms: Optional[str] = None
    quality_clauses: Optional[str] = None
    currency: Optional[str] = "USD"
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None
    line_items: List[LineItem] = Field(default_factory=list)

# =============== Models (Vision JSON) ===============
class Box(BaseModel):
    left: float
    top: float
    right: float
    bottom: float

class Chunk(BaseModel):
    id: str
    type: Literal["text", "header", "table", "section", "other"]
    box: Box
    page: Optional[int] = None  
    text: Optional[str] = None
    level: Optional[int] = None
    rows: Optional[List[List[str]]] = None  
    title: Optional[str] = None
    anchor_id: Optional[str] = None
    label: Optional[str] = None

class VisionCombined(BaseModel):
    markdown: str
    chunks: List[Chunk] 

class VisionMarkdownOnly(BaseModel):
    markdown: str

# =============== Tracing / Notify (no-ops you can replace) ===============
class NoopTracer:
    def start(self, step: str, kind: str, payload: Any = None, meta: dict | None = None):
        return {"step": step, "kind": kind, "payload": payload, "meta": meta or {}}
    def finish(self, ev, output: Any = None): return

class NoopCtx:
    context: Dict[str, Any] = {}
    def __init__(self, **kwargs): self.context = dict(kwargs)

def _notify(ctx: RunContextWrapper | None, message: str, **data):
    try:
        cb = ctx.context.get("notify") if ctx else None
        (cb and callable(cb)) and cb({"message": message, **data})
    except Exception:
        pass

Tracer = NoopTracer  # type: ignore

# =============== Helpers ===============
PAGE_DELIM = "\n\n---\n\n"
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-5-mini")
TEXT_MODEL   = os.getenv("LLM_MODEL", "gpt-5-mini")

SYSTEM_PROMPT_PO = (
    "You are a strict PO parser. Output MUST conform to the PurchaseOrder schema. "
    "Always set the 'page' field to the provided PageIndex. If a value is not on this page, leave it null."
)

VISION_PROMPT_MD = (
    "Convert this document page into clean Markdown.\n"
    "- Preserve visible text faithfully\n"
    "- Use headings (#, ##, ###) for hierarchy\n"
    "- Convert tables to markdown tables\n"
    "- Keep monetary amounts, part numbers, quantities exact\n"
    "Output ONLY the Markdown. No commentary."
)

VISION_PROMPT_COMBINED = (
    "Convert this page to markdown AND identify structural elements.\n"
    "Return strict JSON with the following schema (no extra keys):\n"
    "{\n"
    '  "markdown": "…",\n'
    '  "chunks": [\n'
    '    {\n'
    '      "id": "string",\n'
    '      "type": "text|header|table|section|other",\n'
    '      "box": {"left":0..1, "top":0..1, "right":0..1, "bottom":0..1},\n'
    '      "page": 0-based integer,\n'
    '      // For type="text" or "header": "text": "…"\n'
    '      // For type="header":        "level": 1..6\n'
    '      // For type="table":         "rows": [[cell, ...], ...]\n'
    '      // For type="section":       "title": "…", "anchor_id": "…"?\n'
    '      // For type="other":         "label": "…"?\n'
    '    }\n'
    '  ]\n'
    "}\n"
    "Use normalized coordinates in [0,1]. Return JSON only."
)


def _mk_id(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+","-", (text or "").lower()).strip("-")[:48]
    digest = hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:6]
    return f"{slug}-{digest}" if slug else f"id-{digest}"

def _add_heading_anchors(md: str) -> str:
    out = []
    for line in (md or "").splitlines():
        if line.startswith("#"):
            txt = re.sub(r"^#+\s*", "", line)
            out.append(f'{line} <a id="{_mk_id(txt)}"></a>')
        else:
            out.append(line)
    return "\n".join(out)

def _split_pages_from_combined(combined_text: str) -> List[str]:
    return combined_text.split(PAGE_DELIM) if PAGE_DELIM in combined_text else [combined_text]

def _to_base64_jpeg(img: Image.Image, max_side: int = 1400, quality: int = 65) -> str:
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    bio = io.BytesIO()
    img.save(bio, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(bio.getvalue()).decode("utf-8")

# --- FITZ-BASED CHUNKING (no vision required) -------------------------------

def _normalize_box(x0, y0, x1, y1, pw, ph) -> Box:
    # Clamp and normalize to [0,1]
    left   = max(0.0, min(1.0, x0 / pw if pw else 0.0))
    right  = max(0.0, min(1.0, x1 / pw if pw else 0.0))
    top    = max(0.0, min(1.0, y0 / ph if ph else 0.0))
    bottom = max(0.0, min(1.0, y1 / ph if ph else 0.0))
    # Ensure left<=right & top<=bottom even if input is noisy
    if left > right: left, right = right, left
    if top > bottom: top, bottom = bottom, top
    return Box(left=left, top=top, right=right, bottom=bottom)

def _group_lines_fitz(page, *, y_tol: float = 3.0, top_max: float | None = None, right_max: float | None = None):
    """
    Y-group then left-to-right within each row, similar to pn_extractor.
    Returns tuples: (combined_text, (x0,y0,x1,y1))
    """
    text_dict = page.get_text("dict")
    rows = {}
    for block in text_dict.get("blocks", []):
        if "lines" not in block: 
            continue
        for line in block["lines"]:
            x0, y0, x1, y1 = line["bbox"]
            if top_max is not None and y0 > top_max: 
                continue
            if right_max is not None and x0 > right_max: 
                continue

            line_txt = "".join(span.get("text", "") for span in line.get("spans", []))
            if not line_txt.strip():
                continue

            y_key = round(y0 / y_tol) * y_tol
            rows.setdefault(y_key, []).append((x0, line_txt.strip(), (x0, y0, x1, y1)))

    for y_key in sorted(rows.keys(), reverse=True):  # bottom-up like your agent
        items = sorted(rows[y_key], key=lambda r: r[0])  # left→right
        combined = " ".join(t for _, t, _ in items)
        boxes = [b for *_, b in items]
        x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
        yield combined, (x0, y0, x1, y1)

def fitz_chunks_from_pdf(
    pdf_bytes: bytes,
    *,
    skip_top_first_page: float = 0.20,   # match pn_extractor.SKIP_TOP_FIRST_PAGE
    use_left_width: float = 0.75,        # match pn_extractor.USE_LEFT_WIDTH
    require_digits: bool = False,        # set True to mimic contains_numbers filter
) -> list[Chunk]:
    """
    Build JSON chunks with normalized coords using fitz (no vision).
    Produces Chunk(type='text') compatible with your Vision JSON schema.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is not installed.")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks: list[Chunk] = []
    order = 0

    try:
        for pidx in range(len(doc)):  # 0-based
            page = doc[pidx]
            pw, ph = page.rect.width, page.rect.height

            top_max = ph * (1 - skip_top_first_page) if pidx == 0 else ph
            right_max = pw * use_left_width

            for text, (x0, y0, x1, y1) in _group_lines_fitz(page, top_max=top_max, right_max=right_max):
                if require_digits and not any(c.isdigit() for c in text):
                    continue

                box = _normalize_box(x0, y0, x1, y1, pw, ph)
                # keep id stable-ish for anchors/crossrefs
                cid = _mk_id(f"{pidx}:{order}:{text[:80]}")
                chunks.append(Chunk(
                    id=cid,
                    type="text",
                    box=box,
                    page=pidx,          # 0-based, matching your vision normalization
                    text=text
                ))
                order += 1
    finally:
        doc.close()

    return chunks

def _build_windows(items: List[str], window: int, stride: int) -> List[Tuple[int, int, str]]:
    """
    Return list of (start_idx_1based, end_idx_1based, combined_text) windows.
    """
    n = len(items)
    window = max(1, int(window or 1))
    stride = max(1, int(stride or 1))
    out: List[Tuple[int,int,str]] = []
    i = 0
    while i < n:
        j = min(n, i + window)
        # 1-based page indices for prompts / schema
        start_1b, end_1b = i + 1, j
        combined = PAGE_DELIM.join(items[i:j])
        out.append((start_1b, end_1b, combined))
        if j == n:
            break
        i += stride
    return out

async def _extract_po_window(
    start_page_1b: int,
    end_page_1b: int,
    window_text: str,
    sem: asyncio.Semaphore,
    model: str,
    tracer: Any = None,
    ctx: Any = None
) -> PurchaseOrder:
    async with sem:
        _notify(ctx, "po_agent_start", page_range=f"{start_page_1b}-{end_page_1b}")
        agent = Agent(
            name="PO Window Extractor",
            instructions=SYSTEM_PROMPT_PO,
            model=model,
            output_type=PurchaseOrder,
        )
        # IMPORTANT: keep PageIndex = first page so schema’s 'page' stays stable
        res = await Runner.run(agent, input=(
            f"PageIndex: {start_page_1b}\n"
            f"PageWindow: {start_page_1b}-{end_page_1b}\n\n"
            f"TEXT:\n{window_text}"
        ))
        po: PurchaseOrder = res.final_output
        if po.page is None:
            po.page = start_page_1b
        _notify(ctx, "po_agent_finish", page_range=f"{start_page_1b}-{end_page_1b}", lines=len(po.line_items or []))
        return po


# =============== Text extraction (fast) ===============
def extract_text_from_pdf(pdf_bytes: bytes, tracer: Any, ctx: Any) -> str:
    _notify(ctx, "text_extraction_start")
    ev = tracer.start("text_extraction", "tool_call", payload={"method": "pdfplumber"})
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            t0 = time.perf_counter()
            parts = []
            for i, page in enumerate(pdf.pages, start=1):
                _notify(ctx, "page_extract_start", page=i)
                text = page.extract_text() or ""
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        if table and len(table) > 0:
                            md_table = "\n| " + " | ".join(str(cell or "") for cell in table[0]) + " |\n"
                            md_table += "| " + " | ".join(["---"] * len(table[0])) + " |\n"
                            for row in table[1:]:
                                md_table += "| " + " | ".join(str(cell or "") for cell in row) + " |\n"
                            text += f"\n\n{md_table}\n\n"
                parts.append(text)
                dt1 = time.perf_counter() - t0
                _notify(ctx, f"page_extract_finish in {dt1} seconds", page=i, chars=len(text))
            combined = PAGE_DELIM.join(parts)
            tracer.finish(ev, output={"pages": len(parts), "chars": len(combined)})
            dt2 = time.perf_counter() - t0
            _notify(ctx, f"text_extraction_finish in {dt2} seconds", pages=len(parts))
            return combined
    except Exception as e:
        tracer.finish(ev, output={"error": str(e)})
        raise

# =============== PO extraction (page-parallel with Agents) ===============
async def _extract_po_one_page(page_idx: int, text: str, sem: asyncio.Semaphore, model: str, tracer: Any = None, ctx: Any = None) -> PurchaseOrder:
    async with sem:
        _notify(ctx, "po_agent_start", page=page_idx)
        agent = Agent(
            name="PO Page Extractor",
            instructions=SYSTEM_PROMPT_PO,
            model=model,
            output_type=PurchaseOrder,
        )
        res = await Runner.run(agent, input=f"PageIndex: {page_idx}\n\nTEXT:\n{text}")
        po: PurchaseOrder = res.final_output
        if po.page is None:
            po.page = page_idx
        _notify(ctx, "po_agent_finish", page=page_idx, lines=len(po.line_items or []))
        return po

import datetime as _dt

_DASHES = r"\--–—_"  # ascii -, non-breaking hyphen, en/em dashes, underscore

def _norm_part(p: str | None, *, strip_dashes: bool = True) -> str:
    s = (p or "").upper().strip()
    if strip_dashes:
        s = re.sub(rf"[{_DASHES}\s]+", "", s)  # drop dashes/underscores/whitespace
    else:
        s = re.sub(r"\s+", "", s)              # just collapse whitespace
    return s

def _parse_date(d: str | None) -> _dt.date | None:
    if not d: 
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d-%b-%Y"):
        try:
            return _dt.datetime.strptime(d.strip(), fmt).date()
        except Exception:
            continue
    return None

def _li_score(li: LineItem) -> tuple[int, int, int]:
    # 1) count how many informative fields are present
    filled = sum(bool(x not in (None, "")) for x in [
        li.line_no, li.revision, li.description, li.qty, li.unit_price,
        li.extended_price, li.delivery_date, li.dpas_rating, li.anchor_id
    ])
    # 2) prefer items that have both qty and unit_price
    has_pricing = int(li.qty is not None and li.unit_price is not None)
    # 3) prefer *earlier* delivery dates (smaller ordinal wins)
    d = _parse_date(li.delivery_date)
    date_key = -(d.toordinal()) if d else 0   # earlier => larger score
    return (filled, has_pricing, date_key)


def _reduce_po_pages(per_page: List[PurchaseOrder]) -> PurchaseOrder:
    per_page = sorted(per_page, key=lambda p: p.page or 0)
    merged = PurchaseOrder()
    header_fields = [
        "po_number","buyer","supplier_name","supplier_address",
        "create_date","payment_terms","incoterms","currency",
        "subtotal","tax","total"
    ]
    for p in per_page:
        for f in header_fields:
            if getattr(merged, f) is None:
                setattr(merged, f, getattr(p, f))
    seen = set()
    for p in per_page:
        for li in p.line_items:
            # key = (li.line_no, li.part, li.description, li.delivery_date, li.unit_price, li.qty)
            key = (li.part, li.revision)
            if key in seen: continue
            seen.add(key)
            merged.line_items.append(li)
    merged.page = None
    return merged

# async def po_from_text_pages(pdf_bytes: bytes, *, max_concurrency: int, model: str, tracer: Any, ctx: Any) -> PurchaseOrder:
#     combined = extract_text_from_pdf(pdf_bytes, tracer, ctx)
#     page_texts = _split_pages_from_combined(combined)
#     sem = asyncio.Semaphore(max_concurrency)
#     tasks = [_extract_po_one_page(i+1, t, sem, model, tracer, ctx) for i, t in enumerate(page_texts)]
#     per_page = await asyncio.gather(*tasks)
#     return _reduce_po_pages(per_page)

async def po_from_text_pages(
    pdf_bytes: bytes,
    *,
    max_concurrency: int,
    model: str,
    tracer: Any,
    ctx: Any,
    window_pages: int = 1,
    window_stride: int = 1,
) -> PurchaseOrder:
    combined = extract_text_from_pdf(pdf_bytes, tracer, ctx)
    page_texts = _split_pages_from_combined(combined)

    windows = _build_windows(page_texts, window_pages, window_stride)
    sem = asyncio.Semaphore(max_concurrency)

    tasks = [
        _extract_po_window(s, e, txt, sem, model, tracer, ctx)
        for (s, e, txt) in windows
    ]
    per_window = await asyncio.gather(*tasks)
    return _reduce_po_pages(per_window)


# =============== Vision (page-parallel with Agents) ===============
async def _vision_one_page_markdown(img: Image.Image, page_idx: int, sem: asyncio.Semaphore, model: str, tracer: Any = None, ctx: Any = None) -> str:
    async with sem:
        _notify(ctx, "vision_md_start", page=page_idx)
        # encode JPEG off the event loop
        loop = asyncio.get_running_loop()
        b64 = await loop.run_in_executor(ThreadPoolExecutor(max_workers=8), _to_base64_jpeg, img, 1400, 65)
        agent = Agent(
            name="Vision Markdown",
            instructions=VISION_PROMPT_MD,
            model=model,
            output_type=VisionMarkdownOnly,
        )
        res = await Runner.run(agent, 
                        input=[
            Message(
                content=[
                    ResponseInputTextParam(text=VISION_PROMPT_MD, type="input_text"),
                    ResponseInputImageParam(detail="low", image_url=f"data:image/jpeg;base64,{b64}", type="input_image")
                ],
                role="user"
            )
        ])
        md: VisionMarkdownOnly = res.final_output
        _notify(ctx, "vision_md_finish", page=page_idx, chars=len(md.markdown))
        return md

async def _vision_one_page_combined(img: Image.Image, page_idx: int, sem: asyncio.Semaphore, model: str, tracer: Any = None, ctx: Any = None) -> VisionCombined:
    async with sem:
        _notify(ctx, "vision_json_start", page=page_idx)
        loop = asyncio.get_running_loop()
        b64 = await loop.run_in_executor(ThreadPoolExecutor(max_workers=8), _to_base64_jpeg, img, 1400, 65)
        agent = Agent(
            name="Vision Markdown+Chunks",
            instructions=VISION_PROMPT_COMBINED,
            model=model,
            output_type=VisionCombined,
        )
        res = await Runner.run(agent, 
                        input=[
            Message(
                content=[
                    ResponseInputTextParam(text=VISION_PROMPT_COMBINED, type="input_text"),
                    ResponseInputImageParam(detail="low", image_url=f"data:image/jpeg;base64,{b64}", type="input_image")
                ],
                role="user"
            )
        ])
        vc: VisionCombined = res.final_output
        
        for ch in vc.chunks:
            ch.page = page_idx - 1

            ch.box.left   = max(0.0, min(1.0, ch.box.left))
            ch.box.top    = max(0.0, min(1.0, ch.box.top))
            ch.box.right  = max(0.0, min(1.0, ch.box.right))
            ch.box.bottom = max(0.0, min(1.0, ch.box.bottom))

            if ch.type == "section" and not ch.anchor_id and ch.title:
                ch.anchor_id = _mk_id(ch.title)
        _notify(ctx, "vision_json_finish", page=page_idx, chars=len(vc.markdown), chunks=len(vc.chunks))
        return vc

async def vision_markdown_and_chunks(
    pdf_bytes: bytes,
    *,
    max_concurrency: int,
    model: str,
    want_chunks: bool,
    tracer: Any,
    ctx: Any,
) -> Tuple[str, List[Chunk]]:
    _notify(ctx, "pdf_render_start")
    pages = convert_from_bytes(pdf_bytes, dpi=120)  # lower DPI for speed
    _notify(ctx, "pdf_render_finish", pages=len(pages))

    sem = asyncio.Semaphore(max_concurrency)
    if want_chunks:
        tasks = [_vision_one_page_combined(img, i+1, sem, model, tracer, ctx) for i, img in enumerate(pages)]
        results: List[VisionCombined] = await asyncio.gather(*tasks)
        parts = [r.markdown for r in results]
        chunks: List[Chunk] = []
        for r in results: chunks.extend(r.chunks)
    else:
        tasks = [_vision_one_page_markdown(img, i+1, sem, model, tracer, ctx) for i, img in enumerate(pages)]
        results: List[VisionMarkdownOnly] = await asyncio.gather(*tasks)
        parts = [r.markdown for r in results]
        chunks = []
    combined_md = PAGE_DELIM.join(parts)
    return combined_md, chunks

# =============== Public: concurrent orchestration ===============
async def process_po_file_async(
    file_path: str,
    *,
    do_po: bool = True,
    do_markdown: bool = False,
    do_json_coords: bool = False,
    use_fitz_chunks: bool = False,
    use_vision: bool = False,             
    max_concurrency: int = 200,
    po_window_pages: int = 1,
    po_window_stride: int = 1,
    output_dir: Optional[str] = None,
    model_text: str = TEXT_MODEL,
    model_vision: str = VISION_MODEL,
    tracer: Any = None,
    notify=None,
) -> Dict[str, Any]:
    """
    End-to-end concurrent pipeline. Toggle outputs via args.
    - Always returns a dict with any results that were requested.
    """
    tracer = tracer or NoopTracer()
    ctx = NoopCtx(notify=notify)

    pdf_bytes = open(file_path, "rb").read()
    base = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = output_dir or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # Spawn tasks concurrently
    tasks = {}
    if do_po:
        tasks["po"] = asyncio.create_task(
            po_from_text_pages(
                pdf_bytes,
                max_concurrency=max_concurrency,
                model=model_text,
                tracer=tracer,
                ctx=ctx,
                window_pages=po_window_pages,     
                window_stride=po_window_stride,   
            )
        )

    fitz_chunks: List[Chunk] | None = None
    if do_json_coords and use_fitz_chunks:
        _notify(ctx, "fitz_chunks_start")
        try:
            fitz_chunks = fitz_chunks_from_pdf(pdf_bytes, require_digits=False)
        finally:
            _notify(ctx, "fitz_chunks_finish", chunks=len(fitz_chunks or []))

    if use_vision or (do_markdown and not use_fitz_chunks) or (do_json_coords and not use_fitz_chunks):
        # vision drives markdown/json; set want_chunks if JSON coords requested
        want_chunks = bool(do_json_coords)
        tasks["vision"] = asyncio.create_task(vision_markdown_and_chunks(
            pdf_bytes,
            max_concurrency=max_concurrency,
            model=model_vision,
            want_chunks=want_chunks,
            tracer=tracer,
            ctx=ctx,
        ))

    # Wait for all enabled tasks
    results = {}
    if not tasks:
        return {"error": "Nothing to do. Enable one of do_po/do_markdown/do_json_coords/use_vision."}

    done = await asyncio.gather(*tasks.values())

    # If we used fitz for chunks, assemble markdown + JSON without vision
    if use_fitz_chunks and do_json_coords:
        # Markdown via text extractor (already adds tables); add anchors as you do for vision
        combined_text = extract_text_from_pdf(pdf_bytes, tracer, ctx)
        combined_md = _add_heading_anchors(combined_text)
        results["markdown"] = combined_md
        results["chunks"] = [c.model_dump() for c in (fitz_chunks or [])]

        if do_markdown:
            md_path = os.path.join(output_dir, f"{base}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(combined_md)
            results["markdown_path"] = md_path
            results["markdown_chars"] = len(combined_md)

        # Persist JSON in the same shape as the Vision path
        json_path = os.path.join(output_dir, f"{base}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"markdown": combined_md, "chunks": results["chunks"]}, f, ensure_ascii=False, indent=2)
        results["json_path"] = json_path
        results["chunks_count"] = len(fitz_chunks or [])


    name_map = list(tasks.keys())
    for key, value in zip(name_map, done):
        results[key] = value

    # Persist markdown / JSON if requested (and if vision ran)
    if "vision" in results:
        combined_md, chunks = results["vision"]
        # ID anchors
        combined_md = _add_heading_anchors(combined_md)

        if do_markdown:
            md_path = os.path.join(output_dir, f"{base}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(combined_md)
            results["markdown_path"] = md_path
            results["markdown_chars"] = len(combined_md)

        if do_json_coords:
            json_path = os.path.join(output_dir, f"{base}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"markdown": combined_md, "chunks": [c.model_dump() for c in chunks]}, f, ensure_ascii=False, indent=2)
            results["json_path"] = json_path
            results["chunks_count"] = len(chunks)

        # also return in-memory values for immediate use
        results["markdown"] = combined_md
        results["chunks"] = [c.model_dump() for c in chunks]

    if "po" in results:
        results["purchase_order"] = results["po"].model_dump()
        del results["po"]

    return results