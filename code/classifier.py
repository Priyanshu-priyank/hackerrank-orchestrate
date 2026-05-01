class Classifier:
    HIGH_RISK_KEYWORDS = [
        "fraud",
        "unauthorized",
        "stolen",
        "hacked",
        "compromised",
        "lawsuit",
        "legal action",
        "billing dispute",
        "chargeback",
        "charge back",
        "account deleted",
        "security breach",
        "identity theft",
    ]

    COMPANY_KEYWORDS = {
        "hackerrank": [
            "hackerrank",
            "test",
            "candidate",
            "screen",
            "coding challenge",
            "assessment",
            "recruiter",
        ],
        "claude": ["claude", "conversation", "ai assistant", "prompt", "workspace"],
        "visa": ["visa", "card", "transaction", "payment", "atm", "bank"],
    }

    PROMPT_INJECTION_PATTERNS = [
        "ignore previous instructions",
        "ignore all instructions",
        "disregard",
        "pretend you are",
        "you are now",
        "new instructions",
    ]

    def infer_company(self, issue, subject):
        text = f"{issue or ''} {subject or ''}".lower()
        scores = {
            company: sum(1 for keyword in keywords if keyword in text)
            for company, keywords in self.COMPANY_KEYWORDS.items()
        }
        best_score = max(scores.values()) if scores else 0
        if best_score == 0:
            return "unknown"
        winners = [company for company, score in scores.items() if score == best_score]
        return winners[0] if len(winners) == 1 else "unknown"

    def is_high_risk(self, issue):
        text = (issue or "").lower()
        return any(keyword in text for keyword in self.HIGH_RISK_KEYWORDS)

    def is_prompt_injection(self, issue):
        text = (issue or "").lower()
        return any(pattern in text for pattern in self.PROMPT_INJECTION_PATTERNS)

    def is_invalid(self, issue):
        text = "" if issue is None else str(issue).strip()
        return not text or len(text) < 5
