import os
import random
import pandas as pd

from config import (
    POSITIVE_PAIRS_V1,
    NEGATIVE_PAIRS_V1,
)

from data_pipeline import load_training_data

from blocking import (
    build_block_index,
    get_block_candidates,
)


# ---------------------------------------------------------
# 1. Parse ground truth
# ---------------------------------------------------------

def build_ground_truth_lookup(ground_truth):
    """
    Creates:

        source1_entity_id
            -> set of matched Source2/Source3 entity IDs
    """

    ground_truth_lookup = {}

    for _, row in ground_truth.iterrows():

        source1_id = row["source1_entity_id"]
        matched_ids = row["matched_entity_ids"]

        if pd.isna(matched_ids) or str(matched_ids).strip() == "":
            ground_truth_lookup[source1_id] = set()

        else:
            matched_set = {
                x.strip()
                for x in str(matched_ids).split(",")
                if x.strip()
            }

            ground_truth_lookup[source1_id] = matched_set

    return ground_truth_lookup


# ---------------------------------------------------------
# 2. Generate positive pairs
# ---------------------------------------------------------

def generate_positive_pairs(
    ground_truth,
    max_positive_pairs,
):
    """
    Creates positive training pairs from the ground truth.

    Each positive pair has:

        source1_entity_id
        candidate_entity_id
        label = 1
    """

    positive_pairs = []

    for _, row in ground_truth.iterrows():

        source1_id = row["source1_entity_id"]
        matched_ids = row["matched_entity_ids"]

        if pd.isna(matched_ids):
            continue

        matched_ids = str(matched_ids).strip()

        if not matched_ids:
            continue

        for candidate_id in matched_ids.split(","):

            candidate_id = candidate_id.strip()

            if not candidate_id:
                continue

            positive_pairs.append(
                {
                    "source1_entity_id": source1_id,
                    "candidate_entity_id": candidate_id,
                    "label": 1,
                }
            )

    # Shuffle so that the selected subset is not biased
    random.shuffle(positive_pairs)

    if len(positive_pairs) > max_positive_pairs:
        positive_pairs = positive_pairs[:max_positive_pairs]

    return positive_pairs


# ---------------------------------------------------------
# 3. Generate hard negative pairs
# ---------------------------------------------------------

def generate_hard_negative_pairs(
    source1,
    candidates,
    ground_truth_lookup,
    block_indexes,
    max_negative_pairs,
):
    """
    Generate hard negatives using the blocking system.

    A hard negative is a candidate that:

        - appears in at least one blocking bucket
        - is NOT a true match
        - therefore looks somewhat similar but is incorrect
    """

    negative_pairs = []

    # Shuffle Source 1 entities so that we don't always
    # start with the same entities.
    source1_indices = list(range(len(source1)))

    random.shuffle(source1_indices)

    for source1_index in source1_indices:

        if len(negative_pairs) >= max_negative_pairs:
            break

        source1_row = source1.iloc[source1_index]

        source1_id = source1_row["entity_id"]

        # True matches for this Source1 entity
        true_matches = ground_truth_lookup.get(
            source1_id,
            set(),
        )

        # Get candidates from blocking
        candidate_indices = get_block_candidates(
            source1_row,
            block_indexes,
        )

        if not candidate_indices:
            continue

        # Remove true matches
        hard_negative_indices = []

        for candidate_index in candidate_indices:

            candidate_id = candidates.iloc[
                candidate_index
            ]["entity_id"]

            if candidate_id not in true_matches:
                hard_negative_indices.append(
                    candidate_index
                )

        if not hard_negative_indices:
            continue

        # Shuffle hard negatives
        random.shuffle(hard_negative_indices)

        # Usually one or a few hard negatives per entity
        # is enough for V1.
        for candidate_index in hard_negative_indices:

            candidate_id = candidates.iloc[
                candidate_index
            ]["entity_id"]

            negative_pairs.append(
                {
                    "source1_entity_id": source1_id,
                    "candidate_entity_id": candidate_id,
                    "label": 0,
                }
            )

            if len(negative_pairs) >= max_negative_pairs:
                break

    return negative_pairs


# ---------------------------------------------------------
# 4. Main
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("GENERATING TRAINING PAIRS")
    print("=" * 60)

    # -----------------------------------------------------
    # Load + preprocess through the centralized pipeline
    # -----------------------------------------------------

    print("\n[1/5] Loading and preprocessing data...")

    source1, source2, source3, ground_truth = (
        load_training_data()
    )

    print(f"Source 1: {len(source1):,}")
    print(f"Source 2: {len(source2):,}")
    print(f"Source 3: {len(source3):,}")
    print(f"Ground truth: {len(ground_truth):,}")

    # -----------------------------------------------------
    # Combine Source 2 and Source 3
    # -----------------------------------------------------

    print("\n[2/5] Combining candidate sources...")

    candidates = pd.concat(
        [source2, source3],
        ignore_index=True,
    )

    print(
        f"Total candidate entities: "
        f"{len(candidates):,}"
    )

    # -----------------------------------------------------
    # Build ground truth lookup
    # -----------------------------------------------------

    print("\n[3/5] Building ground truth lookup...")

    ground_truth_lookup = build_ground_truth_lookup(
        ground_truth
    )

    print(
        f"Ground truth entities: "
        f"{len(ground_truth_lookup):,}"
    )

    # -----------------------------------------------------
    # Build blocking indexes
    # -----------------------------------------------------

    print("\n[4/5] Building blocking indexes...")

    block_indexes = build_block_index(
        candidates
    )

    print("Blocking indexes ready.")

    # -----------------------------------------------------
    # Positive pairs
    # -----------------------------------------------------

    print("\nGenerating positive pairs...")

    positive_pairs = generate_positive_pairs(
        ground_truth=ground_truth,
        max_positive_pairs=POSITIVE_PAIRS_V1,
    )

    print(
        f"Positive pairs generated: "
        f"{len(positive_pairs):,}"
    )

    # -----------------------------------------------------
    # Hard negative pairs
    # -----------------------------------------------------

    print("\nGenerating hard negative pairs...")

    negative_pairs = generate_hard_negative_pairs(
        source1=source1,
        candidates=candidates,
        ground_truth_lookup=ground_truth_lookup,
        block_indexes=block_indexes,
        max_negative_pairs=NEGATIVE_PAIRS_V1,
    )

    print(
        f"Hard negative pairs generated: "
        f"{len(negative_pairs):,}"
    )

    # -----------------------------------------------------
    # Combine pairs
    # -----------------------------------------------------

    print("\nCombining training pairs...")

    all_pairs = positive_pairs + negative_pairs

    random.shuffle(all_pairs)

    pairs_df = pd.DataFrame(all_pairs)

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output_dir = "data"

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    output_path = os.path.join(
        output_dir,
        "training_pairs_v1.tsv",
    )

    pairs_df.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING PAIR GENERATION COMPLETE")
    print("=" * 60)

    print(f"\nOutput file:")
    print(output_path)

    print(
        f"\nTotal pairs: "
        f"{len(pairs_df):,}"
    )

    if len(pairs_df) > 0:

        print(
            f"Positive pairs: "
            f"{(pairs_df['label'] == 1).sum():,}"
        )

        print(
            f"Negative pairs: "
            f"{(pairs_df['label'] == 0).sum():,}"
        )

        print("\nSample pairs:")
        print(
            pairs_df.head(10).to_string(
                index=False
            )
        )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()