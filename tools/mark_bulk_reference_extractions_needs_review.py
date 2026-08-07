#!/usr/bin/env python3
"""Correct bulk extraction status without changing PDFs or hashes."""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for meta_path in (ROOT / "docs/papers/ref").glob("*/metadata.json"):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["conversion_status"] = "needs-review"
    conversion = meta.setdefault("conversion", {})
    warnings = conversion.setdefault("warnings", [])
    msg = "Bulk layout-preserving extraction was not visually verified for every equation, table, and section ordering."
    if msg not in warnings:
        warnings.append(msg)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paper_path = meta_path.parent / "paper.md"
    paper = paper_path.read_text(encoding="utf-8")
    paper, count = re.subn(r'(?m)^conversion_status: "[^"]+"$', 'conversion_status: "needs-review"', paper, count=1)
    if count != 1:
        raise RuntimeError(f"frontmatter conversion_status missing or duplicated: {paper_path}")
    paper_path.write_text(paper, encoding="utf-8")
