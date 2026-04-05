"""Memory context formatting for LLM prompts."""


def build_memory_context(memories: list[dict]) -> str:
    """Format memory dicts as a context block. Returns empty string if none."""
    if not memories:
        return ""

    lines = ["Bekannte Informationen über den Haushalt:"]
    for m in memories:
        text = m.get("text", "")
        if not text:
            subject = m.get("subject", "")
            fact = m.get("fact", "")
            if subject and fact:
                text = f"{subject}: {fact}"
        if text:
            lines.append(f"- {text}")

    if len(lines) == 1:
        return ""

    return "\n".join(lines)
