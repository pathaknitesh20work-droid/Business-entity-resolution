import pandas as pd

from normalization import (
    normalize_name,
    normalize_address,
    normalize_country
)


def normalize_dataframe(df):
    """
    Apply the project's standard normalization rules
    to an entity DataFrame.

    The input DataFrame must contain:
        business_name
        business_address
        country

    The returned DataFrame contains:
        norm_name
        norm_address
        norm_country
    """

    df = df.copy()

    # --------------------------------------------------------
    # Business name
    # --------------------------------------------------------

    df["norm_name"] = (
        df["business_name"]
        .fillna("")
        .astype(str)
        .map(normalize_name)
    )

    # --------------------------------------------------------
    # Business address
    # --------------------------------------------------------

    df["norm_address"] = (
        df["business_address"]
        .fillna("")
        .astype(str)
        .map(normalize_address)
    )

    # --------------------------------------------------------
    # Country
    # --------------------------------------------------------

    df["norm_country"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .map(normalize_country)
    )

    return df


def normalize_entity_data(
    source1,
    source2,
    source3
):
    """
    Normalize all three entity sources using exactly
    the same preprocessing pipeline.
    """

    source1 = normalize_dataframe(source1)

    source2 = normalize_dataframe(source2)

    source3 = normalize_dataframe(source3)

    return source1, source2, source3