import pandas as pd

from config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
)

def load_tsv(file_path):
    """
    Load a TSV file into a pandas DataFrame.
    """
    return pd.read_csv(file_path, sep="\t")


def load_training_data():
    """
    Load raw (unnormalized) training TSVs.

    Returns:
        source1, source2, source3, ground_truth
    """

    source1 = load_tsv(TRAIN_SOURCE1)
    source2 = load_tsv(TRAIN_SOURCE2)
    source3 = load_tsv(TRAIN_SOURCE3)
    ground_truth = load_tsv(TRAIN_GROUND_TRUTH)

    return source1, source2, source3, ground_truth


def load_test_data():
    """
    Load raw (unnormalized) test TSVs.

    Returns:
        source1, source2, source3
    """

    source1 = load_tsv(TEST_SOURCE1)
    source2 = load_tsv(TEST_SOURCE2)
    source3 = load_tsv(TEST_SOURCE3)

    return source1, source2, source3