# guardrail.py
import re

# Sensitive information keywords/patterns
SENSITIVE_PATTERNS = [
    # Passwords / credentials
    r"\bpassword\b",
    r"\bpasswd\b",
    r"\bpasscode\b",
    r"\blogin credentials?\b",
    r"\bcredentials?\b",

    # API / authentication secrets
    r"\bapi[\s_-]?key\b",
    r"\bapi[\s_-]?secret\b",
    r"\baccess[\s_-]?token\b",
    r"\bauth[\s_-]?token\b",
    r"\bauthentication[\s_-]?token\b",
    r"\bsecret[\s_-]?key\b",
    r"\bclient[\s_-]?secret\b",
    r"\bprivate[\s_-]?key\b",

    # General secrets
    r"\bsecret\b",
    r"\btoken\b",
    r"\bcredentials?\b",
    r"\bsecurity[\s_-]?key\b",

    # Common sensitive files
    r"\.env\b",
    r"\bid_rsa\b",
    r"\bprivate\.key\b",
    r"\bsecrets?\.json\b",

    # Requests to reveal sensitive information
    r"\breveal.*(password|key|token|secret|credential)",
    r"\bshow.*(password|key|token|secret|credential)",
    r"\bgive me.*(password|key|token|secret|credential)",
    r"\bprovide.*(password|key|token|secret|credential)",
    r"\bextract.*(password|key|token|secret|credential)",
    r"\bfind.*(password|key|token|secret|credential)",
    r"\bprint.*(password|key|token|secret|credential)",
]

COMPILED_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in SENSITIVE_PATTERNS
]


def check_sensitive_request(query: str):
    """
    Returns:
        (True, "")     -> request is allowed
        (False, reason) -> request should be blocked
    """

    if not query or not query.strip():
        return False, "Please enter a question."

    for pattern in COMPILED_PATTERNS:
        if pattern.search(query):
            return (
                False,
                "🚫 Request blocked: I can't provide, retrieve, "
                "or expose passwords, API keys, tokens, secrets, "
                "credentials, or private keys."
            )

    return True, ""


def guardrail(query: str):
    """
    Main guardrail function.
    """

    allowed, message = check_sensitive_request(query)

    if not allowed:
        return False, message

    return True, ""
