from data_loader import load_training_data
from collections import Counter


print("=" * 70)
print("LOADING DATA")
print("=" * 70)

source1, source2, source3, ground_truth = load_training_data()

print("Source 1:", len(source1))
print("Source 2:", len(source2))
print("Source 3:", len(source3))
print("Ground truth:", len(ground_truth))


# ============================================================
# PARSE MATCHED IDS
# ============================================================

def parse_matches(value):
    if value is None:
        return []

    if isinstance(value, float):
        return []

    value = str(value).strip()

    if value == "" or value.lower() == "nan":
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


match_counts = ground_truth["matched_entity_ids"].apply(
    lambda x: len(parse_matches(x))
)


# ============================================================
# BASIC DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("MATCH COUNT DISTRIBUTION")
print("=" * 70)

print(match_counts.describe())


print("\nNumber of S1 entities with each match count:")

distribution = match_counts.value_counts().sort_index()

for count, number in distribution.items():
    print(f"{count:>5} matches : {number:,} entities")


# ============================================================
# ZERO MATCHES
# ============================================================

zero_matches = (match_counts == 0).sum()

print("\n" + "=" * 70)
print("ZERO MATCH ENTITIES")
print("=" * 70)

print("Zero-match S1 entities:", f"{zero_matches:,}")
print(
    "Percentage:",
    f"{zero_matches / len(ground_truth) * 100:.2f}%"
)


# ============================================================
# NON-ZERO MATCHES
# ============================================================

nonzero_matches = (match_counts > 0).sum()

print("\n" + "=" * 70)
print("NON-ZERO MATCH ENTITIES")
print("=" * 70)

print("Entities with at least one match:", f"{nonzero_matches:,}")
print(
    "Percentage:",
    f"{nonzero_matches / len(ground_truth) * 100:.2f}%"
)


# ============================================================
# TOTAL MATCHES
# ============================================================

total_matches = match_counts.sum()

print("\n" + "=" * 70)
print("TOTAL MATCHES")
print("=" * 70)

print("Total matched entity pairs:", f"{total_matches:,}")


# ============================================================
# SOURCE 2 vs SOURCE 3
# ============================================================

print("\n" + "=" * 70)
print("MATCH SOURCE DISTRIBUTION")
print("=" * 70)

source2_ids = set(source2["entity_id"])
source3_ids = set(source3["entity_id"])

s2_matches = 0
s3_matches = 0
unknown_matches = 0

for value in ground_truth["matched_entity_ids"]:

    matches = parse_matches(value)

    for entity_id in matches:

        if entity_id in source2_ids:
            s2_matches += 1

        elif entity_id in source3_ids:
            s3_matches += 1

        else:
            unknown_matches += 1


print("Matches from Source 2:", f"{s2_matches:,}")
print("Matches from Source 3:", f"{s3_matches:,}")
print("Unknown IDs:", f"{unknown_matches:,}")


# ============================================================
# SAMPLE GROUND TRUTH
# ============================================================

print("\n" + "=" * 70)
print("GROUND TRUTH EXAMPLES")
print("=" * 70)

for i in range(10):

    row = ground_truth.iloc[i]

    print("\nSource 1:", row["source1_entity_id"])
    print("Matches:", parse_matches(row["matched_entity_ids"]))


print("\n" + "=" * 70)
print("GROUND TRUTH ANALYSIS COMPLETE")
print("=" * 70)