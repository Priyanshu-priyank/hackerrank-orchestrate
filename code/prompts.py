SYSTEM_PROMPT = """You are a support triage agent for HackerRank, Claude, and Visa.

Rules:
1. Use ONLY the provided corpus excerpts. Never use outside knowledge.
2. Escalate for fraud, stolen card/account, security breach, legal threats, system outages, billing disputes needing identity verification, and zero corpus coverage on sensitive issues.
3. Reply without escalation for FAQs, how-to questions, feature questions, and out-of-scope questions.
4. For out-of-scope or invalid tickets, set status=replied, request_type=invalid, and response=\"I'm sorry, this question is outside the scope of my support capabilities.\"
5. Never fabricate policies, steps, links, or contact numbers.
6. If company is unknown, infer it from the issue content.
7. product_area must be one of: billing, account, screen, community, privacy, travel_support, general_support, conversation_management, out_of_scope, general.
8. Use 2-6 sentences for FAQ answers and bullet steps for how-to answers.
9. If escalating, response must be an empty string \"\".
"""


USER_MESSAGE_TEMPLATE = """Company: {company}
Subject: {subject}
Issue: {issue}

HIGH RISK FLAGS: {flags}

CORPUS EXCERPTS:
{chunks}"""


def build_user_message(company, subject, issue, flags, chunks):
    return USER_MESSAGE_TEMPLATE.format(
        company=company or "unknown",
        subject=subject or "",
        issue=issue or "",
        flags=", ".join(flags) if flags else "none",
        chunks=_format_chunks(chunks),
    )


def _format_chunks(chunks):
    if not chunks:
        return "No relevant corpus excerpts were retrieved."
    formatted = []
    for i, chunk in enumerate(chunks, start=1):
        formatted.append(f"[{i}]\n{chunk}")
    return "\n\n".join(formatted)

