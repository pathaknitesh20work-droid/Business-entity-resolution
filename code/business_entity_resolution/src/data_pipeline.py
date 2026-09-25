from data_loader import load_training_data, load_test_data
from preprocessing import normalize_dataframe


def get_training_data():
    source1, source2, source3, ground_truth = load_training_data()

    source1, source2, source3 = preprocess_training_data(
        source1,
        source2,
        source3,
    )

    return source1, source2, source3, ground_truth


def get_test_data():
    source1, source2, source3 = load_test_data()

    source1, source2, source3 = preprocess_test_data(
        source1,
        source2,
        source3,
    )

    return source1, source2, source3