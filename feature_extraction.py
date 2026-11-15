"""
feature_extraction.py
---------------------
Extracts URL-based features used by the phishing detection model.
All features are derived purely from the URL string itself — no live
DNS lookups or external API calls are required, so the system works
completely offline.

Feature list (16 features):
 1.  url_length           – total character length of the URL
 2.  hostname_length      – length of the hostname portion
 3.  path_length          – length of the URL path
 4.  num_dots             – count of '.' characters in the full URL
 5.  num_hyphens          – count of '-' characters
 6.  num_at              – count of '@' characters (phishing indicator)
 7.  num_question_marks   – count of '?' characters
 8.  num_equal            – count of '=' characters
 9.  num_slashes          – count of '/' characters in the full URL
10.  num_percent          – count of '%' characters (URL encoding abuse)
11.  has_https            – 1 if scheme is https, else 0
12.  has_ip               – 1 if hostname looks like an IPv4 address
13.  has_suspicious_tld   – 1 if TLD is in a known-suspicious list
14.  num_subdomains       – number of sub-domain labels (dots in hostname - 1)
15.  has_phishing_keyword – 1 if URL contains a common phishing keyword
16.  double_slash_in_path – 1 if '//' appears after the scheme portion
"""

import re
import urllib.parse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq",        # free Freenom TLDs heavily abused
    "xyz", "pw", "cc", "top", "click",
    "link", "work", "online", "live",
    "support", "loan", "win", "racing",
    "review", "accountant", "trade",
    "stream", "download", "country",
    "science", "party",
}

PHISHING_KEYWORDS = [
    "login", "signin", "verify", "account", "secure",
    "update", "confirm", "banking", "paypal", "ebay",
    "amazon", "apple", "microsoft", "google", "netflix",
    "password", "credential", "wallet", "free", "prize",
    "winner", "click", "alert", "suspend", "unusual",
    "activity", "limited", "expire", "urgent",
]

# Regex for IPv4 addresses
_IPV4_RE = re.compile(
    r"^(\d{1,3}\.){3}\d{1,3}$"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(url: str) -> dict:
    """
    Given a raw URL string, return a dictionary of numeric features.

    Parameters
    ----------
    url : str
        The URL to analyse (may or may not include the scheme).

    Returns
    -------
    dict
        Mapping of feature name → numeric value (int or float).
    """
    # Normalise: add scheme if missing so urlparse works correctly
    if not url.startswith(("http://", "https://", "ftp://")):
        normalised = "http://" + url
    else:
        normalised = url

    parsed = urllib.parse.urlparse(normalised)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    full_url = url.lower()

    # --- basic counts ---
    url_length = len(url)
    hostname_length = len(hostname)
    path_length = len(path)
    num_dots = full_url.count(".")
    num_hyphens = full_url.count("-")
    num_at = full_url.count("@")
    num_question_marks = full_url.count("?")
    num_equal = full_url.count("=")
    num_slashes = full_url.count("/")
    num_percent = full_url.count("%")

    # --- binary / categorical ---
    has_https = 1 if parsed.scheme == "https" else 0

    has_ip = 1 if _IPV4_RE.match(hostname) else 0

    tld = hostname.rsplit(".", 1)[-1] if "." in hostname else ""
    has_suspicious_tld = 1 if tld.lower() in SUSPICIOUS_TLDS else 0

    # subdomains: "www.bank.example.com" → hostname has 3 dots → 2 subdomains
    hostname_parts = hostname.split(".")
    num_subdomains = max(0, len(hostname_parts) - 2)

    has_phishing_keyword = int(
        any(kw in full_url for kw in PHISHING_KEYWORDS)
    )

    # Double-slash AFTER the scheme (e.g. http://example.com//redirect)
    after_scheme = full_url[full_url.find("//") + 2:]  # skip the scheme's //
    double_slash_in_path = 1 if "//" in after_scheme else 0

    return {
        "url_length": url_length,
        "hostname_length": hostname_length,
        "path_length": path_length,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_at": num_at,
        "num_question_marks": num_question_marks,
        "num_equal": num_equal,
        "num_slashes": num_slashes,
        "num_percent": num_percent,
        "has_https": has_https,
        "has_ip": has_ip,
        "has_suspicious_tld": has_suspicious_tld,
        "num_subdomains": num_subdomains,
        "has_phishing_keyword": has_phishing_keyword,
        "double_slash_in_path": double_slash_in_path,
    }


def extract_feature_vector(url: str) -> list:
    """
    Return feature values as an ordered list (matches FEATURE_NAMES order).
    Use this when passing directly into a trained sklearn model.
    """
    feat = extract_features(url)
    return [feat[name] for name in FEATURE_NAMES]


# Ordered list of feature names — must match the column order used at
# training time (see train_model.py).
FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_at",
    "num_question_marks",
    "num_equal",
    "num_slashes",
    "num_percent",
    "has_https",
    "has_ip",
    "has_suspicious_tld",
    "num_subdomains",
    "has_phishing_keyword",
    "double_slash_in_path",
]


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_cases = [
        ("https://www.google.com", "legitimate"),
        ("http://paypal-secure-login.tk/account/verify?user=123", "phishing"),
        ("http://192.168.1.1/update", "phishing"),
        ("https://amazon.com/dp/B08N5KWB9H", "legitimate"),
        ("http://free-prize-winner.xyz/claim?id=abc123", "phishing"),
    ]
    print(f"{'URL':<55} {'Type':<12} Features")
    print("-" * 100)
    for url, label in test_cases:
        feats = extract_features(url)
        print(f"{url:<55} {label:<12} {feats}")
