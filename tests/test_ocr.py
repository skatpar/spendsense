# Lets "float | None" style type hints work on Python 3.9 too.
from __future__ import annotations

import shutil

import pytest
from PIL import Image, ImageDraw, ImageFont

from src.ocr import extract_date, extract_total, read_receipt

RECEIPT = """IMTIAZ SUPER MARKET
Date: 05/11/2026 18:22
Milk 1L            280.00
Sub Total        1,150.00
GST 0%               0.00
Grand Total      1,150.00
Cash             2,000.00
Change             850.00"""


def test_total_prefers_grand_total_and_skips_subtotal_cash_change():
    assert extract_total(RECEIPT) == 1150.0


@pytest.mark.parametrize("text, expected", [
    ("TOTAL\nRs. 450", 450.0),                      # amount on the next line
    ("Total Items: 3\nNet Payable: 2,340.50", 2340.5),
    ("Subtotal 900\nTax 50\nTotal 950", 950.0),
    ("Thank you for shopping", None),
])
def test_total_variants(text, expected):
    assert extract_total(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("Date: 05/11/2026", "2026-11-05"),
    ("05-11-2026 10:30", "2026-11-05"),
    ("Printed 2026-11-05", "2026-11-05"),
    ("12 Nov 2026", "2026-11-12"),
    ("3 NOVEMBER 2026", "2026-11-03"),
    ("31/02/2026", None),                           # not a real date
    ("no date here", None),
])
def test_date_formats(text, expected):
    assert extract_date(text) == expected


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract is not installed")
def test_read_receipt_on_a_generated_image(tmp_path):
    img = Image.new("RGB", (420, 200), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 20)
    except OSError:
        font = ImageFont.load_default(size=20)
    for i, line in enumerate(["CAMPUS MART", "Date: 05/11/2026", "Grand Total   430.00"]):
        draw.text((20, 20 + i * 50), line, fill="black", font=font)
    path = tmp_path / "receipt.png"
    img.save(path)
    result = read_receipt(path)
    assert set(result) == {"text", "total", "date"}
    assert result["total"] == 430.0
    assert result["date"] == "2026-11-05"
