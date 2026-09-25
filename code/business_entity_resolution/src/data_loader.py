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


def load_training_data():

    s1 = pd.read_csv(TRAIN_SOURCE1, sep="\t")
    s2 = pd.read_csv(TRAIN_SOURCE2, sep="\t")
    s3 = pd.read_csv(TRAIN_SOURCE3, sep="\t")
    gt = pd.read_csv(TRAIN_GROUND_TRUTH, sep="\t")

    return s1, s2, s3, gt


def load_test_data():

    s1 = pd.read_csv(TEST_SOURCE1, sep="\t")
    s2 = pd.read_csv(TEST_SOURCE2, sep="\t")
    s3 = pd.read_csv(TEST_SOURCE3, sep="\t")

    return s1, s2, s3