from data_loader import load_training_data as load_raw_training_data
from data_loader import load_test_data as load_raw_test_data
from preprocessing import normalize_entity_data


def load_training_data():
    source1, source2, source3, ground_truth = load_raw_training_data()

    source1, source2, source3 = normalize_entity_data(
        source1,
        source2,
        source3,
    )

    return source1, source2, source3, ground_truth


def load_test_data():
    source1, source2, source3 = load_raw_test_data()

    source1, source2, source3 = normalize_entity_data(
        source1,
        source2,
        source3,
    )

    return source1, source2, source3