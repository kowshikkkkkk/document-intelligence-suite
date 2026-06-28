"""
Phase 1: Rule-based field extraction from OCR output.

Takes the raw list of (bbox, text, confidence) chunks from EasyOCR and
applies simple pattern-matching heuristics to pull out:
  - date
  - total
  - company (best-effort guess)

This is intentionally simple -- the point of this phase is to establish
a baseline and see where plain rules break down, which then justifies
moving to a trainable model in Phase 2/3.
"""

import re
import sys
from pathlib import Path

# Reuse the OCR function we already wrote
sys.path.append(str(Path(__file__).parent.parent / "data"))
from run_ocr import run_ocr_on_receipt

# Matches formats like 25/12/2018, 25-12-2018, 25.12.2018
DATE_PATTERN = re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b")

# Matches numbers like 9.00, 1,234.50 -- typical currency formatting
AMOUNT_PATTERN = re.compile(r"\b\d{1,3}(?:[,.]\d{3})*\.\d{2}\b")


def extract_date(texts: list[str]) -> str | None:
    for text in texts:
        match = DATE_PATTERN.search(text)
        if match:
            return match.group()
    return None


def extract_total(ocr_results) -> str | None:
    """
    Strategy: find chunks mentioning 'total', then look for the nearest
    amount-shaped number in nearby chunks (since OCR often splits the
    label and the number into separate chunks).
    Prefer chunks mentioning 'rounded total' if present, since that's
    usually the final correct amount on Malaysian-style receipts.
    """
    texts = [text for _, text, _ in ocr_results]

    # First pass: look for "rounded total" specifically
    for i, text in enumerate(texts):
        if "round" in text.lower() and "total" in text.lower():
            amount = _find_amount_near(texts, i)
            if amount:
                return amount

    # Second pass: any line with "total" (but not "sub total")
    for i, text in enumerate(texts):
        if "total" in text.lower() and "sub" not in text.lower():
            amount = _find_amount_near(texts, i)
            if amount:
                return amount

    return None


def _find_amount_near(texts: list[str], index: int, window: int = 3) -> str | None:
    """Look in nearby chunks (same line is often split by OCR) for a number."""
    # Check the same chunk first
    match = AMOUNT_PATTERN.search(texts[index])
    if match:
        return match.group()

    # Then check a small window of nearby chunks
    for offset in range(1, window + 1):
        for i in (index + offset, index - offset):
            if 0 <= i < len(texts):
                match = AMOUNT_PATTERN.search(texts[i])
                if match:
                    return match.group()
    return None


def extract_company(texts: list[str]) -> str | None:
    """
    Heuristic: the company name is usually within the first few chunks,
    and isn't purely numeric/address-like. We just grab the first chunk
    that has at least one letter and isn't too short.
    """
    for text in texts[:5]:
        cleaned = text.strip()
        if len(cleaned) > 4 and any(c.isalpha() for c in cleaned):
            return cleaned
    return None


def extract_fields(receipt_id: str, base_dir=None) -> dict:
    ocr_results = run_ocr_on_receipt(receipt_id, base_dir=base_dir)
    texts = [text for _, text, _ in ocr_results]

    return {
        "company": extract_company(texts),
        "date": extract_date(texts),
        "total": extract_total(ocr_results),
    }


if __name__ == "__main__":
    test_id = "X00016469612"
    fields = extract_fields(test_id)

    print(f"Extracted fields for {test_id}:")
    for key, value in fields.items():
        print(f"  {key:10s}: {value}")