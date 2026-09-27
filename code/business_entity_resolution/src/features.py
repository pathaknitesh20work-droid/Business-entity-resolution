import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# 1. Load the embedding model (call once, reuse everywhere)
# ---------------------------------------------------------

def load_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Default is MiniLM for fast local iteration. Switch to
    Qwen/Qwen3-Embedding-0.6B for the real run on AWS, where
    more CPU/GPU is available.
    """
    return SentenceTransformer(model_name, trust_remote_code=True)


# ---------------------------------------------------------
# 2. Embed all unique entities in a source, once
# ---------------------------------------------------------

def build_embedding_lookup(df, column, embed_model, batch_size=64):
    """
    Returns a dict: entity_id -> embedding vector,
    for the given normalized text column (norm_name or
    norm_address).
    """

    texts = df[column].fillna("").tolist()

    embeddings = embed_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return dict(zip(df["entity_id"], embeddings))


# ---------------------------------------------------------
# 3. Build per-pair features
# ---------------------------------------------------------

def build_features(
    pairs_df,
    source1_df,
    s2s3_df,
    embed_model,
):
    """
    Builds similarity features for every (source1_entity_id,
    candidate_entity_id) pair in pairs_df.

    Only embeds entities that actually appear in pairs_df,
    not the full source1_df/s2s3_df passed in — these can be
    the full normalized frames (millions of rows), while
    pairs_df is bounded by blocking (k candidates per Source1
    entity). Embedding the full frames here was previously
    the actual bottleneck (hours instead of minutes).

    Returns a DataFrame with:
        source1_entity_id
        candidate_entity_id
        name_cos
        addr_cos
        name_fuzz
        addr_fuzz
        name_partial
        country_match
    """

    needed_source1_ids = set(pairs_df["source1_entity_id"].unique())
    needed_candidate_ids = set(pairs_df["candidate_entity_id"].unique())

    source1_subset = source1_df[
        source1_df["entity_id"].isin(needed_source1_ids)
    ].reset_index(drop=True)

    s2s3_subset = s2s3_df[
        s2s3_df["entity_id"].isin(needed_candidate_ids)
    ].reset_index(drop=True)

    print(
        f"[features] Embedding {len(source1_subset)} Source1 + "
        f"{len(s2s3_subset)} candidate entities "
        f"(pool sizes were {len(source1_df)} / {len(s2s3_df)})"
    )

    source1_lookup = source1_subset.set_index("entity_id")
    s2s3_lookup = s2s3_subset.set_index("entity_id")

    name_emb_s1 = build_embedding_lookup(source1_subset, "norm_name", embed_model)
    name_emb_s2s3 = build_embedding_lookup(s2s3_subset, "norm_name", embed_model)
    addr_emb_s1 = build_embedding_lookup(source1_subset, "norm_address", embed_model)
    addr_emb_s2s3 = build_embedding_lookup(s2s3_subset, "norm_address", embed_model)

    rows = []

    for row in pairs_df.itertuples(index=False):
        source1_id = row.source1_entity_id
        candidate_id = row.candidate_entity_id

        s1_row = source1_lookup.loc[source1_id]
        cand_row = s2s3_lookup.loc[candidate_id]

        name_cos = float(np.dot(
            name_emb_s1[source1_id],
            name_emb_s2s3[candidate_id],
        ))

        addr_cos = float(np.dot(
            addr_emb_s1[source1_id],
            addr_emb_s2s3[candidate_id],
        ))

        name_fuzz = fuzz.token_sort_ratio(
            s1_row["norm_name"], cand_row["norm_name"]
        ) / 100

        addr_fuzz = fuzz.token_sort_ratio(
            s1_row["norm_address"], cand_row["norm_address"]
        ) / 100

        name_partial = fuzz.partial_ratio(
            s1_row["norm_name"], cand_row["norm_name"]
        ) / 100

        country_match = int(
            s1_row["norm_country"] == cand_row["norm_country"]
        )

        rows.append(
            {
                "source1_entity_id": source1_id,
                "candidate_entity_id": candidate_id,
                "name_cos": name_cos,
                "addr_cos": addr_cos,
                "name_fuzz": name_fuzz,
                "addr_fuzz": addr_fuzz,
                "name_partial": name_partial,
                "country_match": country_match,
            }
        )

    return pd.DataFrame(rows)


FEATURE_COLUMNS = [
    "name_cos",
    "addr_cos",
    "name_fuzz",
    "addr_fuzz",
    "name_partial",
    "country_match",
]