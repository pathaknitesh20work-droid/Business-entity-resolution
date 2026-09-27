"""
Stratified sampling for the Amazon ML Challenge (Business Entity Resolution).

Produces smaller, representative TRAIN and TEST subsets while trying to
preserve:
  - country distribution (US / India in train; test also has France)
  - the singleton vs matched distribution of Source-1 entities in train
    (singletons are worth a full 1.0 in the F0.5 macro-average, so they
    must not be lost or over/under-sampled)
  - referential integrity: every ID referenced in a sampled ground-truth
    row is guaranteed to exist in the sampled source2/source3 files

IMPORTANT CAVEAT ABOUT THE TEST SET
------------------------------------
The real test set has NO ground truth, so we cannot know which Source-2 /
Source-3 records actually match which Source-1 entities. True match-aware
stratification is therefore impossible for test. This script stratifies
test_source1 by `country` only, and independently samples test_source2 /
test_source3 by their own `country` column at the same rate. This is fine
for smoke-testing your pipeline end-to-end quickly, but it will NOT
preserve recall ceiling the way the train sampling does. For anything
resembling real validation (measuring your own F0.5), use the stratified
TRAIN split produced here, since it comes with ground truth.

Usage
-----
    python3 stratified_sampling.py \
        --data-dir dataset \
        --out-dir dataset_sample \
        --train-frac 0.2 \
        --test-frac 0.2 \
        --seed 42

Expected input layout (relative to --data-dir):
    train/train_source1.tsv
    train/train_source2.tsv
    train/train_source3.tsv
    train/train_ground_truth.tsv
    test/test_source1.tsv
    test/test_source2.tsv
    test/test_source3.tsv

Output layout (relative to --out-dir), same filenames, ready to drop into
your pipeline in place of the full files:
    train/train_source1.tsv
    train/train_source2.tsv
    train/train_source3.tsv
    train/train_ground_truth.tsv
    test/test_source1.tsv
    test/test_source2.tsv
    test/test_source3.tsv
"""

import argparse
import os

import pandas as pd


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return df


def save_tsv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def split_id_list(cell: str):
    """matched_entity_ids / candidate_entity_ids -> list of ids ('' -> [])."""
    cell = (cell or "").strip()
    if not cell:
        return []
    return [x.strip() for x in cell.split(",") if x.strip()]


def stratified_group_sample(df: pd.DataFrame, strat_col: str, frac: float,
                             seed: int) -> pd.DataFrame:
    """
    Sample `frac` of rows from each stratum in strat_col independently, so
    the stratum proportions in the sample match the full data as closely
    as rounding allows. Strata with a single row are kept with probability
    `frac` (via the same .sample mechanics) rather than raising errors, so
    this works fine even on rare strata where sklearn's train_test_split
    would fail.
    """
    if df.empty:
        return df
    parts = []
    for _, group in df.groupby(strat_col, sort=False):
        parts.append(group.sample(frac=frac, random_state=seed))
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]


# --------------------------------------------------------------------------
# TRAIN: match-aware stratified sampling
# --------------------------------------------------------------------------

def sample_train(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame,
                  gt: pd.DataFrame, frac: float, seed: int):
    """
    All four dataframes are passed in already loaded (each file is read
    from disk exactly once by main(); this function does not re-read
    anything), which matters at this dataset's real scale (millions of
    rows per file).
    """
    s1 = s1.copy()
    gt = gt.copy()
    gt["matched_list"] = gt["matched_entity_ids"].apply(split_id_list)
    gt["match_count"] = gt["matched_list"].apply(len)

    # bucket: 0 (singleton), 1, 2+  -- keeps the stratum count small & robust
    def bucket(n):
        if n == 0:
            return "0"
        if n == 1:
            return "1"
        return "2+"

    gt["match_bucket"] = gt["match_count"].apply(bucket)

    s1 = s1.merge(
        gt[["source1_entity_id", "match_bucket", "matched_list"]],
        left_on="entity_id", right_on="source1_entity_id", how="left",
    )
    # entities with no ground-truth row at all are effectively singletons
    s1["match_bucket"] = s1["match_bucket"].fillna("0")
    s1["matched_list"] = s1["matched_list"].apply(
        lambda v: v if isinstance(v, list) else []
    )

    s1["strat_key"] = s1["country"].astype(str) + "|" + s1["match_bucket"]

    s1_sample = stratified_group_sample(s1, "strat_key", frac, seed)
    sampled_s1_ids = set(s1_sample["entity_id"])

    # IDs that MUST be kept: every match of every sampled S1 entity
    required_ids = set()
    for ids in s1_sample["matched_list"]:
        required_ids.update(ids)

    def split_source(prefix_df: pd.DataFrame):
        required = prefix_df[prefix_df["entity_id"].isin(required_ids)]
        rest = prefix_df[~prefix_df["entity_id"].isin(required_ids)]
        rest_sample = stratified_group_sample(rest, "country", frac, seed)
        combined = pd.concat([required, rest_sample], ignore_index=True)
        combined = combined.drop_duplicates(subset="entity_id")
        return combined

    s2_sample = split_source(s2)
    s3_sample = split_source(s3)

    kept_ids = set(s2_sample["entity_id"]) | set(s3_sample["entity_id"])

    # rebuild ground truth for sampled S1 entities, dropping any matched id
    # that didn't survive sampling (shouldn't happen for 'required' ids,
    # but guards against edge cases)
    gt_sample = gt[gt["source1_entity_id"].isin(sampled_s1_ids)].copy()
    gt_sample["matched_list"] = gt_sample["matched_list"].apply(
        lambda ids: [i for i in ids if i in kept_ids]
    )
    gt_sample["matched_entity_ids"] = gt_sample["matched_list"].apply(
        lambda ids: ",".join(ids)
    )
    gt_sample = gt_sample[["source1_entity_id", "matched_entity_ids"]]

    s1_sample = s1_sample.drop(
        columns=["source1_entity_id", "match_bucket", "matched_list", "strat_key"],
        errors="ignore",
    )

    return s1_sample, s2_sample, s3_sample, gt_sample


# --------------------------------------------------------------------------
# TEST: country-only stratified sampling (no ground truth available)
# --------------------------------------------------------------------------

def sample_test(t1: pd.DataFrame, t2: pd.DataFrame, t3: pd.DataFrame,
                 frac: float, seed: int):
    """t1/t2/t3 are already-loaded dataframes; no re-reading from disk."""
    t1_sample = stratified_group_sample(t1, "country", frac, seed)
    t2_sample = stratified_group_sample(t2, "country", frac, seed)
    t3_sample = stratified_group_sample(t3, "country", frac, seed)

    return t1_sample, t2_sample, t3_sample


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def report(name: str, full: pd.DataFrame, sample: pd.DataFrame, strat_col: str):
    print(f"\n{name}: {len(full)} -> {len(sample)} rows")
    if strat_col in full.columns:
        full_dist = full[strat_col].value_counts(normalize=True).round(3)
        sample_dist = sample[strat_col].value_counts(normalize=True).round(3) if len(sample) else sample[strat_col]
        print(f"  {strat_col} proportions (full):   {full_dist.to_dict()}")
        print(f"  {strat_col} proportions (sample): {sample_dist.to_dict()}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="dataset",
                         help="root folder containing train/ and test/ subfolders")
    parser.add_argument("--out-dir", default="dataset_sample",
                         help="where to write the sampled tsv files")
    parser.add_argument("--train-frac", type=float, default=0.2,
                         help="fraction of train Source-1 entities (and proportional S2/S3) to keep")
    parser.add_argument("--test-frac", type=float, default=0.2,
                         help="fraction of test rows to keep, per country, in each source file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=== Loading TRAIN files (each read from disk exactly once) ===")
    s1_full = load_tsv(os.path.join(args.data_dir, "train", "train_source1.tsv"))
    s2_full = load_tsv(os.path.join(args.data_dir, "train", "train_source2.tsv"))
    s3_full = load_tsv(os.path.join(args.data_dir, "train", "train_source3.tsv"))
    gt_full = load_tsv(os.path.join(args.data_dir, "train", "train_ground_truth.tsv"))
    print(f"  source1: {len(s1_full):,} rows, {len(s1_full.columns)} cols {list(s1_full.columns)}")
    print(f"  source2: {len(s2_full):,} rows, {len(s2_full.columns)} cols {list(s2_full.columns)}")
    print(f"  source3: {len(s3_full):,} rows, {len(s3_full.columns)} cols {list(s3_full.columns)}")

    print("\n=== Sampling TRAIN (match-aware, stratified by country x match-bucket) ===")
    s1_sample, s2_sample, s3_sample, gt_sample = sample_train(
        s1_full, s2_full, s3_full, gt_full, args.train_frac, args.seed
    )

    save_tsv(s1_sample, os.path.join(args.out_dir, "train", "train_source1.tsv"))
    save_tsv(s2_sample, os.path.join(args.out_dir, "train", "train_source2.tsv"))
    save_tsv(s3_sample, os.path.join(args.out_dir, "train", "train_source3.tsv"))
    save_tsv(gt_sample, os.path.join(args.out_dir, "train", "train_ground_truth.tsv"))

    report("train_source1", s1_full, s1_sample, "country")
    report("train_source2", s2_full, s2_sample, "country")
    report("train_source3", s3_full, s3_sample, "country")

    full_singleton_mask = (
        s1_full[["entity_id"]]
        .merge(gt_full, left_on="entity_id", right_on="source1_entity_id", how="left")
        ["matched_entity_ids"].fillna("").apply(lambda x: x.strip() == "")
    )
    n_singletons_full = full_singleton_mask.mean()
    n_singletons_sample = gt_sample["matched_entity_ids"].apply(lambda x: x.strip() == "").mean() if len(gt_sample) else 0
    print(f"\n  singleton rate (full):   {n_singletons_full:.3f}")
    print(f"  singleton rate (sample): {n_singletons_sample:.3f}")

    print("\n=== Loading TEST files (each read from disk exactly once) ===")
    t1_full = load_tsv(os.path.join(args.data_dir, "test", "test_source1.tsv"))
    t2_full = load_tsv(os.path.join(args.data_dir, "test", "test_source2.tsv"))
    t3_full = load_tsv(os.path.join(args.data_dir, "test", "test_source3.tsv"))
    print(f"  source1: {len(t1_full):,} rows, {len(t1_full.columns)} cols {list(t1_full.columns)}")
    print(f"  source2: {len(t2_full):,} rows, {len(t2_full.columns)} cols {list(t2_full.columns)}")
    print(f"  source3: {len(t3_full):,} rows, {len(t3_full.columns)} cols {list(t3_full.columns)}")

    print("\n=== Sampling TEST (country-only, no ground truth to stratify on) ===")
    t1_sample, t2_sample, t3_sample = sample_test(t1_full, t2_full, t3_full, args.test_frac, args.seed)

    save_tsv(t1_sample, os.path.join(args.out_dir, "test", "test_source1.tsv"))
    save_tsv(t2_sample, os.path.join(args.out_dir, "test", "test_source2.tsv"))
    save_tsv(t3_sample, os.path.join(args.out_dir, "test", "test_source3.tsv"))

    report("test_source1", t1_full, t1_sample, "country")
    report("test_source2", t2_full, t2_sample, "country")
    report("test_source3", t3_full, t3_sample, "country")

    print(f"\nDone. Sampled files written under: {args.out_dir}/")
    print("Reminder: the test sample has no guaranteed match integrity — "
          "use the train sample (with train_ground_truth.tsv) for your own "
          "F0.5 validation.")


if __name__ == "__main__":
    main()
