import re
import pandas as pd


# =========================================================
# BLOCKING KEY FUNCTIONS
# =========================================================

def first_name_token(text):
    """
    Return the first token of the normalized business name.
    """
    if not text:
        return ""

    parts = text.split()

    return parts[0] if parts else ""


def first_two_name_tokens(text):
    """
    Return the first two tokens of the normalized business name.
    """
    if not text:
        return ""

    parts = text.split()

    return " ".join(parts[:2])


def name_prefix(text, length=4):
    """
    Return the first `length` characters of the normalized
    business name after removing spaces.
    """
    if not text:
        return ""

    text = text.replace(" ", "")

    return text[:length]


def first_address_tokens(text, count=3):
    """
    Return the first `count` tokens of the normalized address.
    """
    if not text:
        return ""

    parts = text.split()

    return " ".join(parts[:count])


def address_number(text):
    """
    Extract the first number appearing in an address.
    """
    if not text:
        return ""

    match = re.search(r"\d+", text)

    if match:
        return match.group()

    return ""


# =========================================================
# CREATE BLOCKING COLUMNS
# =========================================================

def create_blocking_columns(df):
    """
    Create all blocking keys using vectorized pandas operations.

    Expected input columns:

        norm_name
        norm_address
        norm_country
    """

    df = df.copy()

    # -----------------------------------------------------
    # Name keys
    # -----------------------------------------------------

    df["_block_name1"] = (
        df["norm_name"]
        .fillna("")
        .astype(str)
        .str.split()
        .str[0]
        .fillna("")
    )

    df["_block_name2"] = (
        df["norm_name"]
        .fillna("")
        .astype(str)
        .str.split()
        .str[:2]
        .str.join(" ")
    )

    df["_block_prefix"] = (
        df["norm_name"]
        .fillna("")
        .astype(str)
        .str.replace(" ", "", regex=False)
        .str[:4]
    )

    # -----------------------------------------------------
    # Address keys
    # -----------------------------------------------------

    df["_block_address3"] = (
        df["norm_address"]
        .fillna("")
        .astype(str)
        .str.split()
        .str[:3]
        .str.join(" ")
    )

    df["_block_address_number"] = (
        df["norm_address"]
        .fillna("")
        .astype(str)
        .str.extract(
            r"(\d+)",
            expand=False
        )
        .fillna("")
    )

    # -----------------------------------------------------
    # Country
    # -----------------------------------------------------

    df["_block_country"] = (
        df["norm_country"]
        .fillna("")
        .astype(str)
    )

    return df


# =========================================================
# BUILD BLOCK INDEX
# =========================================================

def build_block_index(candidates):
    """
    Build vectorized blocking indexes.

    Each index maps:

        blocking_key -> candidate row indices

    Blocking strategies:

        1. country + first 2 name tokens
        2. country + first name token
        3. country + name prefix
        4. country + first 3 address tokens
        5. country + address number
    """

    candidates = create_blocking_columns(candidates)

    indexes = {}

    # -----------------------------------------------------
    # 1. Country + first 2 name tokens
    # -----------------------------------------------------

    temp = candidates[
        (candidates["_block_country"] != "")
        & (candidates["_block_name2"] != "")
    ].copy()

    temp["_key"] = (
        temp["_block_country"]
        + "|"
        + temp["_block_name2"]
    )

    indexes["country_name2"] = (
        temp.groupby("_key", sort=False)
        .apply(
            lambda x: x.index.tolist(),
            include_groups=False
        )
        .to_dict()
    )

    # -----------------------------------------------------
    # 2. Country + first name token
    # -----------------------------------------------------

    temp = candidates[
        (candidates["_block_country"] != "")
        & (candidates["_block_name1"] != "")
    ].copy()

    temp["_key"] = (
        temp["_block_country"]
        + "|"
        + temp["_block_name1"]
    )

    indexes["country_name1"] = (
        temp.groupby("_key", sort=False)
        .apply(
            lambda x: x.index.tolist(),
            include_groups=False
        )
        .to_dict()
    )

    # -----------------------------------------------------
    # 3. Country + name prefix
    # -----------------------------------------------------

    temp = candidates[
        (candidates["_block_country"] != "")
        & (candidates["_block_prefix"] != "")
    ].copy()

    temp["_key"] = (
        temp["_block_country"]
        + "|"
        + temp["_block_prefix"]
    )

    indexes["country_prefix"] = (
        temp.groupby("_key", sort=False)
        .apply(
            lambda x: x.index.tolist(),
            include_groups=False
        )
        .to_dict()
    )

    # -----------------------------------------------------
    # 4. Country + first 3 address tokens
    # -----------------------------------------------------

    temp = candidates[
        (candidates["_block_country"] != "")
        & (candidates["_block_address3"] != "")
    ].copy()

    temp["_key"] = (
        temp["_block_country"]
        + "|"
        + temp["_block_address3"]
    )

    indexes["country_address3"] = (
        temp.groupby("_key", sort=False)
        .apply(
            lambda x: x.index.tolist(),
            include_groups=False
        )
        .to_dict()
    )

    # -----------------------------------------------------
    # 5. Country + address number
    # -----------------------------------------------------

    temp = candidates[
        (candidates["_block_country"] != "")
        & (candidates["_block_address_number"] != "")
    ].copy()

    temp["_key"] = (
        temp["_block_country"]
        + "|"
        + temp["_block_address_number"]
    )

    indexes["country_address_number"] = (
        temp.groupby("_key", sort=False)
        .apply(
            lambda x: x.index.tolist(),
            include_groups=False
        )
        .to_dict()
    )

    return indexes


# =========================================================
# GET CANDIDATES FOR ONE SOURCE-1 ENTITY
# =========================================================

def get_block_candidates(source_row, indexes):
    """
    Return the union of candidates produced by all blocking
    strategies for one Source-1 entity.
    """

    country = str(
        source_row.get("norm_country", "")
    ).strip()

    name = str(
        source_row.get("norm_name", "")
    ).strip()

    address = str(
        source_row.get("norm_address", "")
    ).strip()

    if not country:
        return set()

    # -----------------------------------------------------
    # Build Source-1 blocking keys
    # -----------------------------------------------------

    name_parts = name.split()

    name1 = (
        name_parts[0]
        if name_parts
        else ""
    )

    name2 = " ".join(
        name_parts[:2]
    )

    prefix = (
        name.replace(" ", "")[:4]
        if name
        else ""
    )

    address_parts = address.split()

    address3 = " ".join(
        address_parts[:3]
    )

    number_match = re.search(
        r"\d+",
        address
    )

    address_num = (
        number_match.group()
        if number_match
        else ""
    )

    # -----------------------------------------------------
    # Collect candidates
    # -----------------------------------------------------

    candidate_indices = set()

    # Country + first 2 name tokens
    if name2:

        key = (
            country
            + "|"
            + name2
        )

        candidate_indices.update(
            indexes["country_name2"].get(
                key,
                []
            )
        )

    # Country + first name token
    if name1:

        key = (
            country
            + "|"
            + name1
        )

        candidate_indices.update(
            indexes["country_name1"].get(
                key,
                []
            )
        )

    # Country + name prefix
    if prefix:

        key = (
            country
            + "|"
            + prefix
        )

        candidate_indices.update(
            indexes["country_prefix"].get(
                key,
                []
            )
        )

    # Country + first 3 address tokens
    if address3:

        key = (
            country
            + "|"
            + address3
        )

        candidate_indices.update(
            indexes["country_address3"].get(
                key,
                []
            )
        )

    # Country + address number
    if address_num:

        key = (
            country
            + "|"
            + address_num
        )

        candidate_indices.update(
            indexes[
                "country_address_number"
            ].get(
                key,
                []
            )
        )

    return candidate_indices