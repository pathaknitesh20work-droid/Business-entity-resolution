import re
import unicodedata


def normalize_text(text):

    if text is None:
        return ""

    text = str(text)

    if text.lower() == "nan":
        return ""

    text = unicodedata.normalize("NFKD", text)

    text = text.lower()

    text = text.replace("&", " and ")

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_name(name):
    return normalize_text(name)


def normalize_address(address):
    return normalize_text(address)


def normalize_country(country):
    return normalize_text(country)