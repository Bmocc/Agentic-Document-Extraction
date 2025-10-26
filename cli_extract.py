from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

from env import load_env, env_str, env_int

# Local import
from agentic_document_extraction import process_po_file_async


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agentic document extraction")
    p.add_argument("pdf", help="Path to input PDF")
    p.add_argument("--out", dest="output_dir", default="./output", help="Output directory")

    # Toggles
    p.add_argument("--po", dest="do_po", action="store_true", help="Extract purchase order JSON")
    p.add_argument("--markdown", dest="do_md", action="store_true", help="Save markdown")
    p.add_argument("--json", dest="do_json", action="store_true", help="Save JSON with coords")
    p.add_argument("--vision", dest="use_vision", action="store_true", help="Use vision for markdown/coords")
    p.add_argument("--fitz", dest="use_fitz", action="store_true", help="Use fitz for coords (no vision)")

    # Models / perf
    p.add_argument("--text-model", dest="text_model", default=None, help="LLM text model id")
    p.add_argument("--vision-model", dest="vision_model", default=None, help="Vision model id")
    p.add_argument("--max-concurrency", dest="max_concurrency", type=int, default=None, help="Max concurrent tasks")
    p.add_argument("--po-window-pages", dest="po_window_pages", type=int, default=1, help="Pages per PO extraction window")
    p.add_argument("--po-window-stride", dest="po_window_stride", type=int, default=1, help="Stride for PO windows")

    return p.parse_args()


def main() -> None:
    load_env()
    args = parse_args()

    # Defaults from env
    text_model = args.text_model or env_str("LLM_MODEL", "gpt-5-mini")
    vision_model = args.vision_model or env_str("VISION_MODEL", "gpt-5-mini")
    max_concurrency = args.max_concurrency or env_int("MAX_CONCURRENCY", 200)

    # Enable at least one action by default if none selected
    do_po = bool(args.do_po)
    do_md = bool(args.do_md)
    do_json = bool(args.do_json)
    use_vision = bool(args.use_vision)
    use_fitz = bool(args.use_fitz)
    if not (do_po or do_md or do_json):
        do_po = True

    out_dir = args.output_dir or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    res: dict[str, Any] = asyncio.run(process_po_file_async(
        file_path=args.pdf,
        do_po=do_po,
        do_markdown=do_md,
        do_json_coords=do_json,
        use_fitz_chunks=use_fitz,
        use_vision=use_vision,
        max_concurrency=max_concurrency,
        po_window_pages=args.po_window_pages,
        po_window_stride=args.po_window_stride,
        output_dir=out_dir,
        model_text=text_model,
        model_vision=vision_model,
    ))

    # Print a short summary
    print("\nExtraction summary:")
    for k in ["purchase_order", "markdown_path", "json_path", "markdown_chars", "chunks_count"]:
        if k in res:
            print(f"- {k}: {res[k] if k != 'purchase_order' else 'OK'}")


if __name__ == "__main__":
    main()

