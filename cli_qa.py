from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any, List

from env import load_env

from agentic_document_query import Chunk, ask_and_annotate_with_highlighter_async


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ask a question and highlight evidence from chunks JSON")
    p.add_argument("pdf", help="Path to the original PDF (for page images)")
    p.add_argument("chunks_json", help="Path to JSON with { markdown, chunks: [...] }")
    p.add_argument("question", help="Question to ask about the document")
    p.add_argument("--out", dest="out_dir", default="./output/highlights", help="Output directory for highlights")
    p.add_argument("--model", dest="qa_model", default=None, help="Text model to use for QA")
    p.add_argument("--dpi", dest="dpi", type=int, default=150, help="DPI for page rendering")
    p.add_argument("--max-chunks", dest="max_chunks", type=int, default=120, help="Max chunks to include in context")
    p.add_argument("--no-answer-box", dest="show_answer", action="store_false", help="Do not overlay the answer text")
    return p.parse_args()


def main() -> None:
    load_env()
    args = parse_args()

    with open(args.chunks_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_chunks: List[dict[str, Any]] = data.get("chunks") or []
    chunks = [Chunk(**c) for c in raw_chunks]

    qa_model = args.qa_model or os.getenv("LLM_MODEL", "gpt-5-mini")

    res = asyncio.run(ask_and_annotate_with_highlighter_async(
        pdf_path=args.pdf,
        question=args.question,
        chunks=chunks,
        qa_model=qa_model,
        out_dir=args.out_dir,
        dpi=args.dpi,
        max_context_chunks=args.max_chunks,
        show_answer=args.show_answer,
    ))

    print("\nAnswer:")
    print(res.get("answer"))
    print("\nHighlights:")
    for p in res.get("highlight_paths", []):
        print("-", p)


if __name__ == "__main__":
    main()

