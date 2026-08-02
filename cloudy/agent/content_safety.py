import re

from langchain.agents.middleware import PIIMiddleware
from langchain.agents.middleware._redaction import (
    PIIDetectionError,
    PIIMatch,
    apply_strategy,
    detect_credit_card,
    detect_email,
    detect_ip,
    detect_mac_address,
    detect_url,
)

from cloudy.observability.logger import get_logger


logger = get_logger(__name__)

# Custom secret patterns — PIIMiddleware's built-in types target classic PII
# (email, IP, ...), not credentials, so these are needed separately. Patterns
# are specific real key formats, not generic "password=" catch-alls, to keep
# the false-positive rate low.
_SECRET_PATTERNS = {
    "aws_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "github_token": re.compile(r"gh[ps]_[a-zA-Z0-9]{36}"),
    "anthropic_key": re.compile(r"sk-ant-[a-zA-Z0-9\-_]{20,}"),
    "openai_key": re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    "private_key_block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def detect_secrets(content: str) -> list[PIIMatch]:
    matches = []
    for label, pattern in _SECRET_PATTERNS.items():
        for m in pattern.finditer(content):
            matches.append(PIIMatch(type=label, value=m.group(), start=m.start(), end=m.end()))
    return matches


def build_pii_middleware() -> list[PIIMiddleware]:
    """One PIIMiddleware instance per detected type, applied to input, output,
    and tool results alike — a secret sitting in indexed code is caught before
    it ever becomes part of the model's reasoning context, not just before
    it's shown to the user. Credit cards are blocked outright rather than
    redacted — there's rarely a legitimate reason to type a real card number
    into a coding assistant at all.
    """
    common = dict(apply_to_input=True, apply_to_output=True, apply_to_tool_results=True)
    return [
        PIIMiddleware("credit_card", strategy="block", **common),
        PIIMiddleware("email", strategy="redact", **common),
        PIIMiddleware("ip", strategy="redact", **common),
        PIIMiddleware("mac_address", strategy="redact", **common),
        PIIMiddleware("url", strategy="redact", **common),
        PIIMiddleware("secret", detector=detect_secrets, strategy="redact", **common),
    ]


def redact_text(text: str) -> tuple[str, list[str]]:
    """Standalone redaction for anything that doesn't go through the agent at
    all — currently just /remember, which writes to long-term memory
    directly, bypassing the agent (and therefore PIIMiddleware) entirely.
    Reuses the exact same detectors the agent path uses, so what counts as
    sensitive is defined once, not duplicated.

    Raises PIIDetectionError if a credit card number is found — same
    block-not-redact behavior as the agent path, and arguably more important
    here, since /remember persists into every future session's system prompt.
    """
    card_matches = detect_credit_card(text)
    if card_matches:
        raise PIIDetectionError("credit_card", card_matches)

    all_matches = (
        detect_email(text) + detect_ip(text) + detect_mac_address(text)
        + detect_url(text) + detect_secrets(text)
    )
    if not all_matches:
        return text, []

    types_found = sorted({m["type"] for m in all_matches})
    sanitized = apply_strategy(text, all_matches, "redact")
    logger.info(f"Redacted from /remember input: {types_found}")
    return sanitized, types_found
