"""Replace vector-PDF labels with proceedings fonts without changing plot data.

Input is an existing selected figure PDF, not a trace or checkpoint. Font files
are read from the local installation and are never copied into the repository.
"""
import argparse
from io import BytesIO
from pathlib import Path
import re

import fitz
from fontTools.ttLib import TTCollection


def math_font_buffer(path):
    collection = TTCollection(path)
    for font in collection.fonts:
        if any(n.toUnicode() == "Cambria Math" for n in font["name"].names):
            stream = BytesIO()
            font.save(stream)
            return stream.getvalue()
    raise ValueError("Cambria Math face not found")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--math-font", type=Path, required=True)
    args = parser.parse_args()
    primary = fitz.Font(fontfile=str(args.font))
    math_buffer = math_font_buffer(args.math_font)
    mathematical = fitz.Font(fontbuffer=math_buffer)
    document = fitz.open(args.source)
    for page in document:
        labels = []
        original_paths = len(page.get_drawings())
        for block in page.get_text("rawdict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    span["text"] = "".join(c["c"] for c in span["chars"])
                    # Matplotlib can draw arrows as vector outlines inside a
                    # text label. Keep their gaps when changing its font widths.
                    if span["text"].startswith("Inference:") and "  " in span["text"]:
                        start = 0
                        for part in span["text"].split("  "):
                            characters = span["chars"][start:start + len(part)]
                            segment = dict(span, text=part, origin=characters[0]["origin"])
                            bounds = fitz.Rect(characters[0]["bbox"])
                            for character in characters[1:]:
                                bounds |= fitz.Rect(character["bbox"])
                            segment["bbox"] = tuple(bounds)
                            labels.append((line["dir"], segment))
                            start += len(part) + 2
                    else:
                        labels.append((line["dir"], span))
                    page.add_redact_annot(span["bbox"], fill=None)
        # Remove text only: retain every vector path and heatmap image.
        page.apply_redactions(images=0, graphics=0)
        assert len(page.get_drawings()) == original_paths
        page.insert_font(fontname="IEIEText", fontfile=str(args.font))
        page.insert_font(fontname="IEIEMath", fontbuffer=math_buffer)
        for direction, span in labels:
            text = re.sub(r"(?<=\d)%", " %", span["text"])
            font = primary
            name = "IEIEText"
            if any(not primary.has_glyph(ord(c)) for c in text if not c.isspace()):
                font, name = mathematical, "IEIEMath"
            if any(not font.has_glyph(ord(c)) for c in text if not c.isspace()):
                raise ValueError(f"Missing glyph: {text!r}")
            rotation = {(1.0, 0.0): 0, (0.0, -1.0): 90,
                        (-1.0, 0.0): 180, (0.0, 1.0): 270}[direction]
            # Preserve each original label's centre and baseline, including
            # rotated axis titles. Plot coordinates and values are untouched.
            width = (span["bbox"][2] - span["bbox"][0] if rotation in (0, 180)
                     else span["bbox"][3] - span["bbox"][1])
            delta = (width - font.text_length(text, fontsize=span["size"])) / 2
            # Multi-span math labels keep their original left edge so nearby
            # superscripts and arrows do not collide after changing typeface.
            anchored = any(c in text for c in "→×α") or text.endswith("D")
            origin = fitz.Point(span["origin"]) + fitz.Point(direction) * (0 if anchored else delta)
            page.insert_text(origin, text, fontname=name, fontsize=span["size"],
                             color=fitz.sRGB_to_pdf(span["color"]), rotate=rotation)
        page.clean_contents()
    document.subset_fonts()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    document.save(args.output, garbage=4, deflate=True)


if __name__ == "__main__":
    main()
