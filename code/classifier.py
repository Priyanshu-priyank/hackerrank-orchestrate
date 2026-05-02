import re
from config import HIGH_RISK_KEYWORDS

def infer_company(issue: str, subject: str) -> str:
    text = (issue + " " + subject).lower()
    
    # Priority matching
    if "hackerrank" in text:
        return "HackerRank"
    if "claude" in text or "anthropic" in text:
        return "Claude"
    if "visa" in text:
        return "Visa"
        
    return "None"

def detect_high_risk(issue: str, subject: str) -> bool:
    text = (issue + " " + subject).lower()
    for keyword in HIGH_RISK_KEYWORDS:
        if keyword in text:
            return True
            
    if "ignore previous instructions" in text or "ignore instructions" in text:
        return True # Flag as risky/invalid
        
    return False

def classify(row) -> dict:
    company = str(row.get('company', 'None'))
    issue = str(row.get('issue', ''))
    subject = str(row.get('subject', ''))
    
    inferred_company = company
    if company.lower() == 'none' or not company.strip():
        inferred_company = infer_company(issue, subject)
        
    is_high_risk = detect_high_risk(issue, subject)
    
    # Check if empty issue
    is_empty = not issue.strip()
    
    return {
        "company": inferred_company,
        "is_high_risk": is_high_risk,
        "is_empty": is_empty
    }
