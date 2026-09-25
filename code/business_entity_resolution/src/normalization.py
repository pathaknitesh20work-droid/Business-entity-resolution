from pathlib import Path
import pandas as pd

from normalization import (
    normalize_name,
    normalize_address,
    normalize_country
)


def load_tsv(file_path):
    """
    Load a TSV file into a pandas DataFrame.
    """

    return pd.read_csv(
        file_path,
        sep="\t"
    )


def clean_dataset(df):
    """
    Clean a business entity dataset.

    Original columns are preserved.
    Normalized columns are added.
    """

    df = df.copy()

    # Normalize business name
    df["business_name_clean"] = (
        df["business_name"]
        .apply(normalize_name)
    )

    # Normalize business address
    df["business_address_clean"] = (
        df["business_address"]
        .apply(normalize_address)
    )

    # Normalize country
    df["country_clean"] = (
        df["country"]
        .apply(normalize_country)
    )

    return df


def load_train_data(data_dir):
    """
    Load and clean the training datasets.

    Returns:
        train_source1
        train_source2
        train_source3
        ground_truth
    """

    data_dir = Path(data_dir)

    train_dir = data_dir / "train"

    # Load training source files
    train_source1 = load_tsv(
        train_dir / "train_source1.tsv"
    )

    train_source2 = load_tsv(
        train_dir / "train_source2.tsv"
    )

    train_source3 = load_tsv(
        train_dir / "train_source3.tsv"
    )

    # Ground truth contains IDs, not business data.
    # Therefore, do not apply business normalization to it.
    ground_truth = load_tsv(
        train_dir / "train_ground_truth.tsv"
    )

    # Clean source datasets
    train_source1 = clean_dataset(train_source1)
    train_source2 = clean_dataset(train_source2)
    train_source3 = clean_dataset(train_source3)

    return (
        train_source1,
        train_source2,
        train_source3,
        ground_truth
    )


def load_test_data(data_dir):
    """
    Load and clean the test datasets.

    Returns:
        test_source1
        test_source2
        test_source3
    """

    data_dir = Path(data_dir)

    test_dir = data_dir / "test"

    # Load test source files
    test_source1 = load_tsv(
        test_dir / "test_source1.tsv"
    )

    test_source2 = load_tsv(
        test_dir / "test_source2.tsv"
    )

    test_source3 = load_tsv(
        test_dir / "test_source3.tsv"
    )

    # Clean source datasets
    test_source1 = clean_dataset(test_source1)
    test_source2 = clean_dataset(test_source2)
    test_source3 = clean_dataset(test_source3)

    return (
        test_source1,
        test_source2,
        test_source3
    )