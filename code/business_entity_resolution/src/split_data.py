import pandas as pd
import numpy as np

from data_loader import load_training_data


# ============================================================
# SETTINGS
# ============================================================

VALIDATION_FRACTION = 0.10
RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training data...")

source1, source2, source3, ground_truth = load_training_data()

print("Source 1:", len(source1))
print("Ground truth:", len(ground_truth))


# ============================================================
# CREATE ENTITY-LEVEL SPLIT
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

indices = np.arange(len(ground_truth))

rng.shuffle(indices)

validation_size = int(
    len(indices) * VALIDATION_FRACTION
)

validation_indices = indices[:validation_size]
train_indices = indices[validation_size:]


train_ground_truth = ground_truth.iloc[train_indices].copy()
validation_ground_truth = ground_truth.iloc[validation_indices].copy()


# ============================================================
# SAVE
# ============================================================

output_dir = "../data/splits"

import os
os.makedirs(output_dir, exist_ok=True)


train_ground_truth.to_csv(
    f"{output_dir}/train_ground_truth.tsv",
    sep="\t",
    index=False
)

validation_ground_truth.to_csv(
    f"{output_dir}/validation_ground_truth.tsv",
    sep="\t",
    index=False
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("SPLIT COMPLETE")
print("=" * 70)

print("Total S1 entities:", len(ground_truth))

print(
    "Training S1 entities:",
    len(train_ground_truth)
)

print(
    "Validation S1 entities:",
    len(validation_ground_truth)
)

print(
    "Training percentage:",
    f"{len(train_ground_truth) / len(ground_truth) * 100:.2f}%"
)

print(
    "Validation percentage:",
    f"{len(validation_ground_truth) / len(ground_truth) * 100:.2f}%"
)

print("\nSplit files saved.")