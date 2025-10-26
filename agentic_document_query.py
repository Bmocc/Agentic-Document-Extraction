from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Literal, Optional
import os, json

from env import load_env
from providers import get_provider
load_env()
PROVIDER = get_provider()

from pydantic import BaseModel, Field, ConfigDict
from collections import defaultdict
from pdf2image import convert_from_bytes
from PIL import Image, ImageDraw, ImageFont

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
    
    level: Optional[int] 
    
    rows: Optional[List[List[str]]] = None  
   
    title: Optional[str] = None
    anchor_id: Optional[str] = None
    
    label: Optional[str] = None

class Citation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    chunk_id: str
    page: int
    box: Box 

class AnswerWithCitations(BaseModel):
    model_config = ConfigDict(extra='forbid')
    answer: str
    citations: List[Citation] = Field(default_factory=list)


def _truncate(s: str, max_chars: int = 600) -> str:
    s = s or ""
    return s if len(s) <= max_chars else (s[:max_chars] + " …")

def _table_preview(rows: List[List[str]], max_rows: int = 12, max_cols: int = 8, max_cell: int = 60) -> List[List[str]]:
    out = []
    for r in rows[:max_rows]:
        row = [(c or "")[:max_cell] for c in r[:max_cols]]
        out.append(row)
    return out

def build_chunk_context_json(chunks: List[Chunk], max_chunks: int = 120) -> str:
    """
    Convert chunks to a compact JSON array the agent can scan. 
    Keeps ordering; you can pre-sort if you prefer.
    """
    entries = []
    for ch in chunks[:max_chunks]:
        entry = {
            "id": ch.id,
            "type": ch.type,
            "page": ch.page if ch.page is not None else 0,
            "box": ch.box.model_dump(),
        }
        if ch.type in ("text", "header") and ch.text:
            entry["text"] = _truncate(ch.text, 600)
            if ch.type == "header" and ch.level is not None:
                entry["level"] = ch.level
        elif ch.type == "section":
            entry["title"] = ch.title or ""
            if ch.anchor_id:
                entry["anchor_id"] = ch.anchor_id
        elif ch.type == "table" and ch.rows:
            entry["rows"] = _table_preview(ch.rows, 12, 8, 60)
        elif ch.type == "other":
            if ch.label: entry["label"] = ch.label
            if ch.text:  entry["text"]  = _truncate(ch.text, 300)
        entries.append(entry)
    return json.dumps({"chunks": entries}, ensure_ascii=False)


QA_INSTRUCTIONS = (
    "You answer questions about the purchase order using ONLY the provided chunk list.\n"
    "Each chunk has: id, type, page, box, and an abbreviated text/table preview when available.\n"
    "Rules:\n"
    "1) If the answer is not in the chunks, say you don't know.\n"
    "2) Return the supporting chunk IDs AND their page+box coordinates in citations.\n"
    "3) Only cite chunks that directly support the answer. Prefer the smallest set.\n"
    "4) Copy page and box from the chunk you cite. Do not make up coordinates.\n"
)

async def _run_qa_agent(question: str, chunks_json: str, model: str = "gpt-5-mini") -> AnswerWithCitations:
    """
    chunks_json: output of build_chunk_context_json(chunks)
    """
    prompt = (
        f"Question:\n{question}\n\n"
        "Below is the JSON array of chunks. Use them to answer and provide citations:\n\n"
        f"{chunks_json}"
    )
    res = await PROVIDER.run_structured_text(
        name="PO QA (Chunk-Cited)",
        instructions=QA_INSTRUCTIONS,
        model=model,
        output_type=AnswerWithCitations,
        input_text=prompt,
    )
    return res


def _chunk_to_element_dict(ch: Chunk) -> Dict[str, Any]:
    bbox = {
        "x": float(ch.box.left),
        "y": float(ch.box.top),
        "width": float(ch.box.right - ch.box.left),
        "height": float(ch.box.bottom - ch.box.top),
    }
    elem: Dict[str, Any] = {
        "id": ch.id,
        "type": ch.type,
        "bbox": bbox,
        "content": {}
    }
    if ch.type in ("text", "header") and getattr(ch, "text", None):
        elem["content"]["text"] = ch.text
        if ch.type == "header" and getattr(ch, "level", None) is not None:
            elem["content"]["level"] = ch.level
    elif ch.type == "section":
        elem["content"]["title"] = ch.title or ""
        if ch.anchor_id:
            elem["content"]["anchor_id"] = ch.anchor_id
    elif ch.type == "table" and ch.rows:
        preview_rows = []
        for r in ch.rows[:8]:
            preview_rows.append([(c or "")[:40] for c in r[:8]])
        elem["content"]["rows"] = preview_rows
        if ch.text: 
            elem["content"]["text"] = ch.text[:120]
    return elem

def _elements_grouped_by_page(chunks: List[Chunk]) -> Dict[int, List[Dict[str, Any]]]:
    by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ch in chunks:
        page = ch.page if ch.page is not None else 0
        by_page[page].append(_chunk_to_element_dict(ch))
    return by_page

def highlight_query_evidence(
    image: Image.Image,
    elements: List[Dict[str, Any]],
    element_ids: List[str],
    output_path: str,
    show_answer: bool = True,
    answer_text: str = ""
) -> None:
    """
    Highlight specific elements that were used as evidence for a query answer.
    
    Args:
        image: Original page image
        elements: All elements on this page
        element_ids: IDs of elements that support the answer
        output_path: Where to save the highlighted image
        show_answer: Whether to show the answer text on the image
        answer_text: The answer to display
    """
    # Create copy with alpha channel for transparency
    img_copy = image.copy().convert('RGBA')
    
    # Create overlay for highlighting
    overlay = Image.new('RGBA', img_copy.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay, 'RGBA')
    
    width, height = image.size
    
    # Load font
    try:
        font_large = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Find matching elements
    highlighted_elements = [e for e in elements if e.get('id') in element_ids]
    
    if not highlighted_elements:
        print(f"Warning: No matching elements found for IDs: {element_ids}")
        # Still save the image
        img_copy.save(output_path)
        return
    
    # Draw all elements as light gray boxes first (context)
    for elem in elements:
        if elem.get('id') not in element_ids:
            bbox = elem.get('bbox', {})
            if bbox:
                x = bbox['x'] * width
                y = bbox['y'] * height
                w = bbox['width'] * width
                h = bbox['height'] * height
                
                # Light gray outline for non-evidence elements
                draw.rectangle(
                    [x, y, x + w, y + h],
                    outline=(200, 200, 200, 150),
                    width=1
                )
    
    # Highlight evidence elements with bright colors
    colors = [
        (255, 0, 0, 120),      # Red
        (0, 255, 0, 120),      # Green
        (255, 165, 0, 120),    # Orange
        (0, 150, 255, 120),    # Blue
        (255, 0, 255, 120),    # Magenta
    ]
    
    for idx, elem in enumerate(highlighted_elements):
        bbox = elem.get('bbox', {})
        if not bbox:
            continue
        
        x = bbox['x'] * width
        y = bbox['y'] * height
        w = bbox['width'] * width
        h = bbox['height'] * height
        
        # Choose color
        color = colors[idx % len(colors)]
        
        # Draw filled rectangle with transparency
        draw.rectangle(
            [x, y, x + w, y + h],
            fill=color,
            outline=(color[0], color[1], color[2], 255),
            width=3
        )
        
        # Draw element label
        label = f"Evidence {idx + 1}: {elem.get('type', 'unknown')}"
        
        # Label background
        label_bbox = draw.textbbox((x, y - 20), label, font=font_small)
        draw.rectangle(
            label_bbox,
            fill=(0, 0, 0, 200)
        )
        
        # Label text
        draw.text(
            (x, y - 20),
            label,
            fill=(255, 255, 255, 255),
            font=font_small
        )
        
        # Extract and show text content if available
        content = elem.get('content', {})
        if isinstance(content, dict):
            # Try to get text content
            text = content.get('text', '')
            if not text and 'headers' in content:
                # For tables, show first row as preview
                text = f"Table: {content.get('headers', [])}"
            
            if text:
                # Truncate if too long
                preview = text[:100] + "..." if len(text) > 100 else text
                
                # Draw text box inside the highlighted area
                text_y = y + 5
                max_width = w - 10
                
                # Word wrap the text
                words = preview.split()
                lines = []
                current_line = []
                
                for word in words:
                    test_line = ' '.join(current_line + [word])
                    bbox_test = draw.textbbox((0, 0), test_line, font=font_small)
                    if bbox_test[2] - bbox_test[0] <= max_width:
                        current_line.append(word)
                    else:
                        if current_line:
                            lines.append(' '.join(current_line))
                        current_line = [word]
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                # Draw text lines with background
                for line in lines[:3]:  # Max 3 lines
                    text_bbox = draw.textbbox((x + 5, text_y), line, font=font_small)
                    # Semi-transparent background for readability
                    draw.rectangle(
                        [text_bbox[0] - 2, text_bbox[1] - 2, 
                         text_bbox[2] + 2, text_bbox[3] + 2],
                        fill=(255, 255, 255, 230)
                    )
                    draw.text((x + 5, text_y), line, fill=(0, 0, 0, 255), font=font_small)
                    text_y += 16
    
    # Composite the overlay onto the image
    img_copy = Image.alpha_composite(img_copy, overlay)
    
    # Add answer text at top if requested
    if show_answer and answer_text:
        # Convert back to RGB to add opaque text box
        img_copy = img_copy.convert('RGB')
        draw_final = ImageDraw.Draw(img_copy)
        
        # Create answer box at top
        answer_lines = []
        words = answer_text.split()
        current_line = []
        max_answer_width = width - 40
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox_test = draw_final.textbbox((0, 0), test_line, font=font_large)
            if bbox_test[2] - bbox_test[0] <= max_answer_width:
                current_line.append(word)
            else:
                if current_line:
                    answer_lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            answer_lines.append(' '.join(current_line))
        
        # Draw answer box
        box_height = len(answer_lines) * 20 + 40
        draw_final.rectangle(
            [10, 10, width - 10, 10 + box_height],
            fill=(0, 100, 200),
            outline=(255, 255, 255),
            width=2
        )
        
        # Draw "ANSWER:" label
        draw_final.text((20, 20), "ANSWER:", fill=(255, 255, 0), font=font_large)
        
        # Draw answer lines
        y_pos = 45
        for line in answer_lines:
            draw_final.text((20, y_pos), line, fill=(255, 255, 255), font=font_large)
            y_pos += 20
    else:
        img_copy = img_copy.convert('RGB')
    
    # Save
    img_copy.save(output_path)
    print(f"Saved highlighted image: {output_path}")

async def ask_and_annotate_with_highlighter_async(
    pdf_path: str,
    question: str,
    *,
    chunks: List[Chunk],
    qa_model: str = "gpt-5-mini",
    out_dir: str = "./output/highlights",
    dpi: int = 150,
    max_context_chunks: int = 120,
    show_answer: bool = True,
) -> Dict[str, Any]:
    """
    - Feeds JSON chunks directly to an Agent (no embeddings).
    - Agent returns answer + citations (chunk_id + page + box).
    - We verify/fix citations against authoritative chunks, render only cited pages,
      and call your highlight_query_evidence(...) to save PNGs.
    """
    assert chunks, "No chunks provided."
    os.makedirs(out_dir, exist_ok=True)

    # 1) Compact context JSON for the Agent
    ctx_json = build_chunk_context_json(chunks, max_chunks=max_context_chunks)

    # 2) Ask Agent for answer + citations
    qa: AnswerWithCitations = await _run_qa_agent(question, ctx_json, model=qa_model)

    # 3) Verify citations: replace page/box from authoritative chunks by ID (avoid model drift)
    id_map = {ch.id: ch for ch in chunks}
    fixed_citations: List[Citation] = []
    for c in qa.citations:
        ch = id_map.get(c.chunk_id)
        if not ch:
            continue
        fixed_citations.append(
            Citation(
                chunk_id=ch.id,
                page=ch.page if ch.page is not None else 0,
                box=ch.box
            )
        )

    # Fallback: if no citations returned, pick the first reasonable chunk by naive heuristic
    if not fixed_citations and chunks:
        first = chunks[0]
        fixed_citations.append(Citation(chunk_id=first.id, page=first.page or 0, box=first.box))

    # 4) Build element sets per page for your highlighter
    elements_by_page = _elements_grouped_by_page(chunks)

    # 5) Render only the cited pages and call your annotator
    pdf_bytes = open(pdf_path, "rb").read()
    all_pages = convert_from_bytes(pdf_bytes, dpi=dpi)  # PIL Images

    # Group citations by page
    cites_by_page: Dict[int, List[str]] = defaultdict(list)
    for c in fixed_citations:
        cites_by_page[c.page].append(c.chunk_id)

    image_paths: List[str] = []
    base = os.path.splitext(os.path.basename(pdf_path))[0]

    for page_idx, ids in cites_by_page.items():
        if not (0 <= page_idx < len(all_pages)):
            continue
        image = all_pages[page_idx]
        elements = elements_by_page.get(page_idx, [])
        out_path = os.path.join(out_dir, f"{base}_qa_p{page_idx+1}.png")

        # **** Use your annotator here ****
        highlight_query_evidence(
            image=image,
            elements=elements,
            element_ids=ids,
            output_path=out_path,
            show_answer=show_answer,
            answer_text=qa.answer,
        )

        image_paths.append(out_path)

    return {
        "answer": qa.answer,
        "citations": [c.model_dump() for c in fixed_citations],
        "highlight_paths": image_paths,
    }
