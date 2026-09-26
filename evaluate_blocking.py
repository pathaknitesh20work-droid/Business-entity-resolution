import pandas as pd
import re
from collections import defaultdict

from data_loader import load_training_data
from normalization import (
    normalize_name,
    normalize_address,
    normalize_country
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_SIZE = 2000
RANDOM_SEED = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

source1, source2, source3, ground_truth = load_training_data()

print("Source 1:", len(source1))
print("Source 2:", len(source2))
print("Source 3:", len(source3))


# ============================================================
# COMBINE SOURCE 2 + SOURCE 3
# ============================================================

candidates = pd.concat(
    [source2, source3],
    ignore_index=True
)

print("\nCombined candidate records:", len(candidates))


# ============================================================
# NORMALIZATION
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZING DATA")
print("=" * 70)


def safe_normalize(value, func):

    if pd.isna(value):
        return ""

    return func(str(value))


source1["norm_name"] = source1["business_name"].map(
    lambda x: safe_normalize(x, normalize_name)
)

source1["norm_address"] = source1["business_address"].map(
    lambda x: safe_normalize(x, normalize_address)
)

source1["norm_country"] = source1["country"].map(
    lambda x: safe_normalize(x, normalize_country)
)


candidates["norm_name"] = candidates["business_name"].map(
    lambda x: safe_normalize(x, normalize_name)
)

candidates["norm_address"] = candidates["business_address"].map(
    lambda x: safe_normalize(x, normalize_address)
)

candidates["norm_country"] = candidates["country"].map(
    lambda x: safe_normalize(x, normalize_country)
)

print("Normalization complete.")


# ============================================================
# BLOCK FUNCTIONS
# ============================================================

def first_name(text):

    if not text:
        return ""

    return text.split()[0]


def first_two_names(text):

    if not text:
        return ""

    tokens = text.split()

    return " ".join(tokens[:2])


def name_prefix(text, length=4):

    if not text:
        return ""

    return text.replace(" ", "")[:length]


def first_address_tokens(text, count=3):

    if not text:
        return ""

    return " ".join(text.split()[:count])


def address_number(text):

    if not text:
        return ""

    match = re.search(r"\b\d+\b", text)

    if match:
        return match.group(0)

    return ""


# ============================================================
# CREATE BLOCK COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("CREATING BLOCK KEYS")
print("=" * 70)


# -------------------------
# Candidate blocks
# -------------------------

candidates["block_country_name2"] = (
    candidates["norm_country"]
    + "||"
    + candidates["norm_name"].map(first_two_names)
)

candidates["block_country_name1"] = (
    candidates["norm_country"]
    + "||"
    + candidates["norm_name"].map(first_name)
)

candidates["block_country_prefix"] = (
    candidates["norm_country"]
    + "||"
    + candidates["norm_name"].map(name_prefix)
)

candidates["block_country_address3"] = (
    candidates["norm_country"]
    + "||"
    + candidates["norm_address"].map(first_address_tokens)
)

candidates["block_country_address_number"] = (
    candidates["norm_country"]
    + "||"
    + candidates["norm_address"].map(address_number)
)

candidates["block_name2"] = (
    candidates["norm_name"].map(first_two_names)
)


# -------------------------
# Source 1 blocks
# -------------------------

source1["block_country_name2"] = (
    source1["norm_country"]
    + "||"
    + source1["norm_name"].map(first_two_names)
)

source1["block_country_name1"] = (
    source1["norm_country"]
    + "||"
    + source1["norm_name"].map(first_name)
)

source1["block_country_prefix"] = (
    source1["norm_country"]
    + "||"
    + source1["norm_name"].map(name_prefix)
)

source1["block_country_address3"] = (
    source1["norm_country"]
    + "||"
    + source1["norm_address"].map(first_address_tokens)
)

source1["block_country_address_number"] = (
    source1["norm_country"]
    + "||"
    + source1["norm_address"].map(address_number)
)

source1["block_name2"] = (
    source1["norm_name"].map(first_two_names)
)


print("Block keys created.")


# ============================================================
# BUILD SET-BASED INDEXES
# ============================================================

print("\n" + "=" * 70)
print("BUILDING BLOCK SETS")
print("=" * 70)


block_columns = [
    "block_country_name2",
    "block_country_name1",
    "block_country_prefix",
    "block_country_address3",
    "block_country_address_number",
    "block_name2"
]


indexes = {}


for column in block_columns:

    print("Building:", column)

    groups = candidates.groupby(column).groups

    indexes[column] = groups


print("\nAll indexes built.")


# ============================================================
# CANDIDATE ENTITY ID LOOKUP
# ============================================================

candidate_ids = candidates["entity_id"].to_numpy()


# ============================================================
# GROUND TRUTH SAMPLE
# ============================================================

validation_sample = ground_truth.sample(
    n=min(SAMPLE_SIZE, len(ground_truth)),
    random_state=RANDOM_SEED
)

source1_index = source1.set_index("entity_id")


# ============================================================
# GROUND TRUTH PARSER
# ============================================================

def parse_matches(value):

    if pd.isna(value):
        return set()

    value = str(value).strip()

    if not value or value.lower() == "nan":
        return set()

    return {
        x.strip()
        for x in value.split(",")
        if x.strip()
    }


# ============================================================
# EVALUATE BLOCK
# ============================================================

def evaluate_block(block_column):

    index = indexes[block_column]

    total_true = 0
    retrieved_true = 0

    entities_with_match = 0
    entities_retrieved = 0

    total_candidates = 0

    for _, gt_row in validation_sample.iterrows():

        true_ids = parse_matches(
            gt_row["matched_entity_ids"]
        )

        # Ignore genuine no-match entities
        if not true_ids:
            continue

        entities_with_match += 1

        s1_id = gt_row["source1_entity_id"]

        s1_row = source1_index.loc[s1_id]

        key = s1_row[block_column]

        # Directly retrieve row positions
        row_positions = index.get(key)

        if row_positions is None:

            candidate_set = set()

        else:

            # Convert only this small block to IDs
            candidate_set = set(
                candidate_ids[row_positions]
            )

        total_candidates += len(candidate_set)

        intersection = true_ids.intersection(
            candidate_set
        )

        total_true += len(true_ids)
        retrieved_true += len(intersection)

        if intersection:
            entities_retrieved += 1


    pair_recall = (
        retrieved_true / total_true
        if total_true
        else 0
    )

    entity_recall = (
        entities_retrieved / entities_with_match
        if entities_with_match
        else 0
    )

    average_candidates = (
        total_candidates / len(validation_sample)
    )


    return (
        pair_recall,
        entity_recall,
        average_candidates
    )


# ============================================================
# RUN EXPERIMENT
# ============================================================

print("\n" + "=" * 70)
print("BLOCKING RECALL EXPERIMENT")
print("=" * 70)

results = []


for block in block_columns:

    print("\nEvaluating:", block)

    pair_recall, entity_recall, avg_candidates = (
        evaluate_block(block)
    )

    results.append(
        (
            block,
            pair_recall,
            entity_recall,
            avg_candidates
        )
    )

    print(
        f"Pair recall:   {pair_recall * 100:.2f}%"
    )

    print(
        f"Entity recall: {entity_recall * 100:.2f}%"
    )

    print(
        f"Avg candidates: {avg_candidates:.2f}"
    )


# ============================================================
# FINAL TABLE
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    f"{'BLOCK':<35}"
    f"{'PAIR RECALL':>15}"
    f"{'ENTITY RECALL':>18}"
    f"{'AVG CANDIDATES':>18}"
)

print("-" * 90)


for block, pair_recall, entity_recall, avg_candidates in results:

    print(
        f"{block:<35}"
        f"{pair_recall * 100:>14.2f}%"
        f"{entity_recall * 100:>17.2f}%"
        f"{avg_candidates:>18.2f}"
    )


print("\n" + "=" * 70)
print("BLOCKING EXPERIMENT COMPLETE")
print("=" * 70)