"""
Data Extractor Module
Extracts structured data from classified documents using regex patterns
"""

import re
from typing import Dict, Any, Optional


def extract_invoice_data(text: str) -> Dict[str, Any]:
    """Extract structured data from invoice documents."""
    data = {
        "invoice_number": None,
        "date": None,
        "company": None,
        "total_amount": None
    }

    # Invoice number patterns
    invoice_patterns = [
        r"Invoice\s*#?\s*:?\s*(\w+-?\d+)",
        r"Invoice\s+Number\s*:?\s*(\w+-?\d+)",
        r"INV[-\s]?(\d+)",
        r"#(\d+)"
    ]
    for pattern in invoice_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data["invoice_number"] = match.group(1) if not match.group(1).startswith("#") else match.group(1)
            break

    # Date patterns
    date_patterns = [
        r"Date\s*:?\s*(\d{4}-\d{2}-\d{2})",
        r"Date\s*:?\s*(\d{2}/\d{2}/\d{4})",
        r"Date\s*:?\s*(\d{2}-\d{2}-\d{4})",
        r"(\d{4}-\d{2}-\d{2})"
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data["date"] = match.group(1)
            break

    # Company name
    company_patterns = [
        r"Company\s*:?\s*(.+?)(?:\n|$)",
        r"From\s*:?\s*(.+?)(?:\n|$)",
        r"Billed\s+by\s*:?\s*(.+?)(?:\n|$)"
    ]
    for pattern in company_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data["company"] = match.group(1).strip()
            break

    # Total amount
    amount_patterns = [
        r"Total\s+Amount\s*:?\s*\$?([\d,]+\.?\d*)",
        r"Total\s*:?\s*\$?([\d,]+\.?\d*)",
        r"Amount\s+Due\s*:?\s*\$?([\d,]+\.?\d*)",
        r"\$\s*([\d,]+\.?\d*)"
    ]
    for pattern in amount_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount_str = match.group(1).replace(",", "")
            try:
                data["total_amount"] = float(amount_str)
            except ValueError:
                pass
            break

    return data


def extract_resume_data(text: str) -> Dict[str, Any]:
    """Extract structured data from resume documents."""
    data = {
        "name": None,
        "email": None,
        "phone": None,
        "experience_years": None
    }

    # Name - usually first line or after "Name:"
    lines = text.split("\n")
    if lines:
        # Check if first line looks like a name (no special chars, 2-4 words)
        first_line = lines[0].strip()
        if re.match(r"^[A-Za-z\s\.]+$", first_line) and 1 <= len(first_line.split()) <= 4:
            data["name"] = first_line

    name_patterns = [
        r"Name\s*:?\s*(.+?)(?:\n|$)",
    ]
    if not data["name"]:
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["name"] = match.group(1).strip()
                break

    # Email
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    match = re.search(email_pattern, text)
    if match:
        data["email"] = match.group(0)

    # Phone
    phone_patterns = [
        r"Phone\s*:?\s*([\+\d\-\(\)\s]+)",
        r"Tel\s*:?\s*([\+\d\-\(\)\s]+)",
        r"(\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})"
    ]
    for pattern in phone_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            phone = match.group(1).strip()
            # Clean up phone number
            if len(re.sub(r"\D", "", phone)) >= 10:
                data["phone"] = phone
                break

    # Experience years
    exp_patterns = [
        r"Experience\s*:?\s*(\d+)\s*years?",
        r"(\d+)\s*years?\s+(?:of\s+)?experience",
        r"(\d+)\+?\s*years?"
    ]
    for pattern in exp_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                data["experience_years"] = int(match.group(1))
            except ValueError:
                pass
            break

    return data


def extract_utility_bill_data(text: str) -> Dict[str, Any]:
    """Extract structured data from utility bill documents."""
    data = {
        "account_number": None,
        "date": None,
        "usage_kwh": None,
        "amount_due": None
    }

    # Account number
    account_patterns = [
        r"Account\s*(?:Number|No\.?|#)?\s*:?\s*([A-Z]*-?\d+)",
        r"Acct\s*#?\s*:?\s*(\w+-?\d+)"
    ]
    for pattern in account_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data["account_number"] = match.group(1)
            break

    # Billing date
    date_patterns = [
        r"Billing\s+Date\s*:?\s*(\d{4}-\d{2}-\d{2})",
        r"Statement\s+Date\s*:?\s*(\d{4}-\d{2}-\d{2})",
        r"Date\s*:?\s*(\d{4}-\d{2}-\d{2})",
        r"(\d{4}-\d{2}-\d{2})"
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data["date"] = match.group(1)
            break

    # Usage kWh
    usage_patterns = [
        r"Usage\s*:?\s*(\d+)\s*kWh",
        r"(\d+)\s*kWh",
        r"Consumption\s*:?\s*(\d+)"
    ]
    for pattern in usage_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                data["usage_kwh"] = int(match.group(1))
            except ValueError:
                pass
            break

    # Amount due
    amount_patterns = [
        r"Amount\s+Due\s*:?\s*\$?([\d,]+\.?\d*)",
        r"Total\s+Due\s*:?\s*\$?([\d,]+\.?\d*)",
        r"Balance\s+Due\s*:?\s*\$?([\d,]+\.?\d*)"
    ]
    for pattern in amount_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount_str = match.group(1).replace(",", "")
            try:
                data["amount_due"] = float(amount_str)
            except ValueError:
                pass
            break

    return data


def extract_data(text: str, document_class: str) -> Dict[str, Any]:
    """Extract structured data based on document classification."""
    if document_class == "Invoice":
        return extract_invoice_data(text)
    elif document_class == "Resume":
        return extract_resume_data(text)
    elif document_class == "Utility Bill":
        return extract_utility_bill_data(text)
    else:
        # Other or Unclassifiable - no extraction
        return {}
