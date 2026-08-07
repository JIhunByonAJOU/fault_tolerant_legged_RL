#!/usr/bin/env python3
"""Prepare a PDF for careful agent-led Markdown conversion using Poppler tools."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_TOOLS = ("pdfinfo", "pdftotext")


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or "command failed"
        fail(f"{' '.join(command[:2])}: {detail}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_pdfinfo(text: str) -> dict[str, str | int | None]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            values[key.strip().lower().replace(" ", "_")] = value.strip()
    pages_raw = values.get("pages")
    pages = int(pages_raw) if pages_raw and pages_raw.isdigit() else None
    return {
        "title": values.get("title") or None,
        "author": values.get("author") or None,
        "subject": values.get("subject") or None,
        "keywords": values.get("keywords") or None,
        "creator": values.get("creator") or None,
        "producer": values.get("producer") or None,
        "creation_date": values.get("creationdate") or None,
        "pages": pages,
        "page_size": values.get("page_size") or None,
        "pdf_version": values.get("pdf_version") or None,
        "encrypted": values.get("encrypted") or None,
    }


def parse_pages(spec: str, maximum: int | None) -> list[int]:
    pages: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
        if not match:
            fail(f"invalid page selection: {token!r}")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start < 1 or end < start:
            fail(f"invalid page range: {token!r}")
        if maximum is not None and end > maximum:
            fail(f"page {end} exceeds document page count {maximum}")
        pages.update(range(start, end + 1))
    return sorted(pages)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="input PDF; never modified")
    parser.add_argument("output_dir", type=Path, help="empty staging directory")
    parser.add_argument(
        "--render-pages",
        metavar="PAGES",
        help="render comma-separated 1-based pages/ranges, e.g. 1,3-5",
    )
    parser.add_argument("--render-dpi", type=int, default=150)
    args = parser.parse_args()

    source = args.pdf.expanduser().resolve()
    if not source.is_file():
        fail(f"PDF not found: {source}")
    with source.open("rb") as stream:
        signature = stream.read(5)
    if signature != b"%PDF-":
        fail(f"file does not start with a PDF signature: {source}")
    for tool in REQUIRED_TOOLS:
        if shutil.which(tool) is None:
            fail(f"required command is unavailable: {tool}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        fail(f"output directory must be empty: {output_dir}")

    info_text = run(["pdfinfo", str(source)]).stdout
    pdf_metadata = parse_pdfinfo(info_text)
    raw_text = output_dir / "raw-layout.txt"
    run(["pdftotext", "-layout", "-enc", "UTF-8", str(source), str(raw_text)])

    rendered: list[str] = []
    if args.render_pages:
        if shutil.which("pdftoppm") is None:
            fail("--render-pages requires pdftoppm")
        image_dir = output_dir / "page-images"
        image_dir.mkdir()
        for page in parse_pages(args.render_pages, pdf_metadata["pages"]):
            prefix = image_dir / f"page-{page:04d}"
            run(
                [
                    "pdftoppm",
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                    "-r",
                    str(args.render_dpi),
                    "-singlefile",
                    "-png",
                    str(source),
                    str(prefix),
                ]
            )
            rendered.append(str(prefix.with_suffix(".png")))

    text = raw_text.read_text(encoding="utf-8", errors="replace")
    manifest = {
        "source_path": str(source),
        "source_name": source.name,
        "sha256": sha256(source),
        "size_bytes": source.stat().st_size,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "pdfinfo": pdf_metadata,
        "extraction": {
            "tool": "pdftotext -layout -enc UTF-8",
            "raw_text_path": str(raw_text),
            "characters": len(text),
            "replacement_characters": text.count("�"),
            "form_feed_page_breaks": text.count("\f"),
            "rendered_pages": rendered,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
