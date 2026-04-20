"""
Extract text from all PDFs in a folder and save them as .txt files.

Input folder:  data/raw
Output folder: data/processed
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


INPUT_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Read all pages from a PDF and return combined text.

    If a page has no extractable text, we safely skip it.
    """
    reader = PdfReader(str(pdf_path))

    parts: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text()
        if not page_text:
            continue
        parts.append(page_text)

    # Join with newlines so pages don't run together.
    return "\n".join(parts).strip()


def pdf_to_txt_path(pdf_path: Path) -> Path:
    """Create the output .txt path using the same base filename."""
    return OUTPUT_DIR / f"{pdf_path.stem}.txt"


def main() -> None:
    # Ensure output folder exists.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all PDF files in the input folder (non-recursive).
    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {INPUT_DIR.resolve()}")
        print("Extraction complete")
        return

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        try:
            text = extract_text_from_pdf(pdf_path)
        except Exception as exc:
            # If something goes wrong for a single file, log it and continue.
            print(f"  Skipped (error): {pdf_path.name} -> {exc}")
            continue

        out_path = pdf_to_txt_path(pdf_path)

        # Write UTF-8 text so it opens cleanly in most editors.
        out_path.write_text(text, encoding="utf-8")

    print("Extraction complete")


if __name__ == "__main__":
    main()

