import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------
# 1. Build a reusable blocking index over candidate entities
# ---------------------------------------------------------

def build_block_index(candidates, k=20, ngram_range=(2, 4)):
    """
    Fits a TF-IDF vectorizer + NearestNeighbors index over the
    normalized names of the candidate pool (Source2 + Source3).

    `candidates` must already be normalized (has norm_name).

    Returns a dict bundling everything needed to query the
    index later, so it only has to be fit once and reused
    across many lookups (per-row hard-negative mining, or
    bulk candidate generation).
    """

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=1,
    )

    candidate_matrix = vectorizer.fit_transform(
        candidates["norm_name"]
    )

    nn_model = NearestNeighbors(
        n_neighbors=min(k, len(candidates)),
        metric="cosine",
    )
    nn_model.fit(candidate_matrix)

    return {
        "vectorizer": vectorizer,
        "nn_model": nn_model,
        "candidates": candidates,
        "k": k,
    }


# ---------------------------------------------------------
# 2. Query the index for a single Source1 row
# ---------------------------------------------------------

def get_block_candidates(source1_row, block_indexes):
    """
    Returns the candidate DataFrame positions (iloc indices
    into block_indexes["candidates"]) nearest to a single
    Source1 row's normalized name.

    Used by generate_pairs.py for per-row hard-negative
    mining.
    """

    vectorizer = block_indexes["vectorizer"]
    nn_model = block_indexes["nn_model"]

    query_vec = vectorizer.transform([source1_row["norm_name"]])

    _, indices = nn_model.kneighbors(query_vec)

    return indices[0].tolist()


# ---------------------------------------------------------
# 3. Bulk candidate generation for candidate_pairs.tsv
# ---------------------------------------------------------

def generate_candidates(source1_df, s2s3_df, k=20):
    """
    Generates the candidate set for every Source1 entity in
    one bulk pass (much faster than calling get_block_candidates
    row-by-row for the full dataset).

    Both inputs must already be normalized (norm_name present).

    Returns a DataFrame with:
        source1_entity_id
        candidate_entity_ids   (list of S2-/S3- entity_id strings)

    This is the direct source for candidate_pairs.tsv, and the
    input to feature building / the matching model.
    """

    block_indexes = build_block_index(s2s3_df, k=k)

    vectorizer = block_indexes["vectorizer"]
    nn_model = block_indexes["nn_model"]

    query_matrix = vectorizer.transform(source1_df["norm_name"])

    _, indices = nn_model.kneighbors(query_matrix)

    rows = []

    for row_position, source1_row in enumerate(
        source1_df.itertuples(index=False)
    ):
        candidate_positions = indices[row_position]

        candidate_ids = s2s3_df.iloc[
            candidate_positions
        ]["entity_id"].tolist()

        rows.append(
            {
                "source1_entity_id": source1_row.entity_id,
                "candidate_entity_ids": candidate_ids,
            }
        )

    return pd.DataFrame(rows)

def flatten_candidates(candidates_df):
    """
    Converts a candidates DataFrame from:

        source1_entity_id | candidate_entity_ids (list)

    into one row per (source1_entity_id, candidate_entity_id)
    pair — the shape feature building and model
    inference need.

    Rows with an empty candidate list are dropped, since
    there's nothing to featurize for them (they'll surface
    later as singletons if the model predicts no matches).
    """

    rows = []

    for _, row in candidates_df.iterrows():
        source1_id = row["source1_entity_id"]
        candidate_ids = row["candidate_entity_ids"]

        for candidate_id in candidate_ids:
            rows.append(
                {
                    "source1_entity_id": source1_id,
                    "candidate_entity_id": candidate_id,
                }
            )

    return pd.DataFrame(
        rows,
        columns=["source1_entity_id", "candidate_entity_id"],
    )