import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# CORS enabled for all origins so a Cloudflare Worker (or any grader) can call this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ExtractRequest(BaseModel):
    invoice_text: str

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

def parse_amount(s):
    if s is None:
        return None
    s = s.replace(",", "").strip()
    try:
        return round(float(s), 2)
    except ValueError:
        return None

def find_invoice_no(text):
    m = re.search(r"(?:Invoice\s*No|Ref)\s*[:\-]?\s*([A-Za-z0-9/\-]+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else None

def find_date(text):
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.search(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b", text)
    if m:
        day, mon_name, year = m.groups()
        mon = MONTHS.get(mon_name.lower())
        if mon:
            return f"{int(year):04d}-{mon:02d}-{int(day):02d}"
    m = re.search(r"\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b", text)
    if m:
        mon_name, day, year = m.groups()
        mon = MONTHS.get(mon_name.lower())
        if mon:
            return f"{int(year):04d}-{mon:02d}-{int(day):02d}"
    return None

def find_vendor(text):
    m = re.search(r"Vendor\s*[:\-]\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    first_line = text.strip().splitlines()[0]
    m2 = re.match(r"(.+?)\s*[—\-]\s*(Tax Invoice|Invoice)", first_line, re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return None

def find_subtotal(text):
    m = re.search(r"Subtotal\s*[.:\-]*\s*Rs\.?\s*([\d,]+\.\d{2})", text, re.IGNORECASE)
    return parse_amount(m.group(1)) if m else None

def find_tax(text):
    m = re.search(r"(?:GST|IGST|CGST|SGST|Tax)\s*\(?\d*%?\)?\s*[.:\-]*\s*Rs\.?\s*([\d,]+\.\d{2})", text, re.IGNORECASE)
    return parse_amount(m.group(1)) if m else None

def find_currency(text):
    m = re.search(r"Currency\s*[:\-]\s*([A-Za-z]{3})", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    if re.search(r"\bRs\.?\b|₹|INR", text):
        return "INR"
    if re.search(r"\$|USD", text):
        return "USD"
    return None

@app.post("/extract")
def extract(req: ExtractRequest):
    text = req.invoice_text
    return {
        "invoice_no": find_invoice_no(text),
        "date": find_date(text),
        "vendor": find_vendor(text),
        "amount": find_subtotal(text),
        "tax": find_tax(text),
        "currency": find_currency(text),
    }
