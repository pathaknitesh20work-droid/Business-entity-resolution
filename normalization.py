import re


LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt", "limited", "ltd",
    "llc", "llp", "pllc", "inc", "incorporated",
    "corporation", "corp", "co",
]

ADDRESS_ABBREVIATIONS = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bapt\b": "apartment",
    r"\bfl\b": "floor",
    r"\bno\b": "number",
}

COUNTRY_ALIASES = {
    "usa": "us",
    "u.s.a": "us",
    "u.s.a.": "us",
    "united states": "us",
    "united states of america": "us",
    "in": "india",
    "fr": "france",
}


def _clean_basic(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_name(name):
    name = _clean_basic(name)
    for suf in LEGAL_SUFFIXES:
        name = re.sub(rf"\b{suf}\b", "", name)
    return re.sub(r"\s+", " ", name).strip()


def normalize_address(address):
    address = _clean_basic(address)
    for pattern, replacement in ADDRESS_ABBREVIATIONS.items():
        address = re.sub(pattern, replacement, address)
    return re.sub(r"\s+", " ", address).strip()


def normalize_country(country):
    country = _clean_basic(country)
    return COUNTRY_ALIASES.get(country, country)