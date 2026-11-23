"""Receipt photo -> {"text", "total", "date"} using the pre-trained Tesseract
OCR engine. We only prepare the image and read the text; nothing is trained."""

# Lets "float | None" style type hints work on Python 3.9 too.
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps

# Tesseract reads best when letters are roughly 30 px tall; phone photos of a
# small receipt are often below that, so we upscale anything narrower than this.
MIN_WIDTH = 1200

# A number like 1,250 or 1250.00 or 1 250. Commas are thousand separators.
NUMBER = r"(\d{1,3}(?:[,\s]\d{3})+|\d+)(?:\.(\d{1,2}))?"

# Checked in this order: a "Grand Total" line beats a plain "Total" line.
TOTAL_KEYWORDS = [r"grand\s*total", r"net\s*payable", r"net\s*total", r"amount\s*due",
                  r"total\s*amount", r"net\s*amount", r"total"]
# Lines that contain "total" or a number but are not the amount paid.
SKIP_WORDS = r"sub\s*-?\s*total|tax|gst|vat|cash|change|discount|tender|qty|items?"

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def load_image(path: str | Path) -> Image.Image:
    """Open a photo the right way up. Phones store rotation as an EXIF tag
    rather than rotating the pixels, and Tesseract ignores that tag."""
    return ImageOps.exif_transpose(Image.open(path))


def preprocess(img: Image.Image) -> np.ndarray:
    """Grayscale -> upscale if small -> light threshold."""
    gray = np.array(img.convert("L"))
    if gray.shape[1] < MIN_WIDTH:
        scale = MIN_WIDTH / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # A small blur removes paper texture so the threshold doesn't turn it into specks.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Adaptive threshold copes with shadows across a photo better than one global cut-off.
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 15)


def is_readable(line: str) -> bool:
    """A readable line contains a real word (2+ letters) and is mostly letters,
    digits and spaces. A lone '.00' or '~;|_' is an OCR fragment, not a line."""
    if not re.search(r"[A-Za-z]{2,}", line):
        return False
    clean_chars = sum(ch.isalnum() or ch == " " for ch in line)
    return clean_chars / len(line) > 0.7


def readability(text: str) -> tuple[int, int]:
    """Score used to compare OCR outputs: first the number of readable lines,
    then (as a tie-break) how many of them keep a price on the same line as
    the item name, which is what the total extractor needs."""
    readable = [ln for ln in text.splitlines() if ln.strip() and is_readable(ln)]
    with_number = [ln for ln in readable if re.search(r"\d", ln)]
    return len(readable), len(with_number)


def run_tesseract(image: np.ndarray) -> str:
    """Try two page layouts and keep whichever reads more cleanly.
    psm 4 = one column of variable-size text, psm 6 = one uniform block."""
    texts = [pytesseract.image_to_string(image, config=f"--psm {psm}") for psm in (4, 6)]
    return max(texts, key=readability)


def parse_number(match: re.Match) -> float:
    whole = re.sub(r"[,\s]", "", match.group(1))
    return float(f"{whole}.{match.group(2) or 0}")


def extract_total(text: str) -> float | None:
    """Find the amount on the best 'total' line. If the line has no number,
    the value is often on the next line, so we look there too."""
    lines = [ln.strip() for ln in text.lower().splitlines() if ln.strip()]
    for keyword in TOTAL_KEYWORDS:
        for i, line in enumerate(lines):
            if not re.search(keyword, line) or re.search(SKIP_WORDS, line):
                continue
            for candidate in (line, lines[i + 1] if i + 1 < len(lines) else ""):
                numbers = list(re.finditer(NUMBER, candidate))
                if numbers:
                    return parse_number(numbers[-1])  # the amount sits at the end
    return None


def _valid_date(year: int, month: int, day: int) -> str | None:
    try:
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def extract_date(text: str) -> str | None:
    """Return the first valid date as 'YYYY-MM-DD'. Supported formats:
    dd/mm/yyyy, dd-mm-yyyy, yyyy-mm-dd and dd Mon yyyy (e.g. 05 Nov 2026)."""
    patterns = [
        (r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", lambda m: (m[1], m[2], m[3])),
        (r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", lambda m: (m[3], m[2], m[1])),
        (r"\b(\d{1,2})[\s-]*([A-Za-z]{3})[A-Za-z]*[\s,.-]*(\d{4})\b",
         lambda m: (m[3], MONTHS.get(m[2].lower()[:3]), m[1])),
    ]
    for pattern, to_ymd in patterns:
        for match in re.finditer(pattern, text):
            year, month, day = to_ymd(match)
            if month is not None and (iso := _valid_date(int(year), int(month), int(day))):
                return iso
    return None


def read_receipt(path: str | Path) -> dict:
    """Full pipeline for one photo. Anything not found is None."""
    text = run_tesseract(preprocess(load_image(path)))
    return {"text": text, "total": extract_total(text), "date": extract_date(text)}
