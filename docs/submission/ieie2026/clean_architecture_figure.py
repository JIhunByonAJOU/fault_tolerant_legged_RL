"""Remove peripheral prose from the original vector architecture figure.

Requires PyMuPDF >= 1.24.2. Retains the original boxes, arrows, dimensions,
fonts and core labels; only the three explanatory lines are redacted.
"""
from pathlib import Path

import fitz


def main():
    assets = Path(__file__).resolve().parent / "private/assets"
    source = assets / "paper_workflow_v8.pdf"
    output = assets / "paper_workflow_clean.pdf"
    document = fitz.open(source)
    assert len(document) == 1
    page = document[0]
    paths = page.get_drawings()
    prefixes = ("Training:", "Critic also receives", "Inference:")
    removed = []
    retained = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"])
            if text.startswith(prefixes):
                page.add_redact_annot(line["bbox"], fill=None)
                removed.append(text)
            else:
                retained.append(text)
    assert len(removed) == 3, removed
    page.apply_redactions(images=0, graphics=0)
    # Removing text shifts paint sequence numbers without changing geometry.
    geometry = lambda drawings: [{k: v for k, v in d.items() if k != "seqno"}
                                 for d in drawings]
    assert geometry(page.get_drawings()) == geometry(paths), "Core vector geometry changed"
    text = page.get_text()
    assert all(label in text for label in retained)
    assert all(label not in text for label in removed)
    # All core paths lie between y=18 and y=104; remove the empty note margins.
    page.set_cropbox(fitz.Rect(0, 15, page.rect.width, 104.5))
    document.set_metadata({})
    document.save(output, garbage=4, deflate=True)
    print(f"{output}: removed {len(removed)} lines; retained {len(paths)} vector paths")


if __name__ == "__main__":
    main()
