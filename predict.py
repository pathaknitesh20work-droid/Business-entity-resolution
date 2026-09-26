import xgboost as xgb 
from features import FEATURE_COLUMNS
def load_model(path):
    model = xbg.Booster()
    model.load_model(str(path))

    threshold_path = str(path).replace(".json","_threshold.txt")
    with open(threshold_path) as f:
        threshold = float(f.read().strip())

    return model,threshold

def predict_matches(model,test_features,threshold):
   # Adds pred_prob and pred)label columns to test)features based on the trained model and tunes threshold.
    dtest = xgb.DMatrix(test_features[FEATURE_COLUMNS])
    probs = model.predict(dtest)

    result = test_features.copy()
    result["pred_prob"] = probs
    result["pred_label"] = (probs > threshold).astype(int)

    return result

## Assemble final output files

def write_candidate_pairs(test_candidates,path):
    ### Wrties candidate_pairs.tsv directly from the blocking output (test_candidates), before any model filtering.
    ### test_candidates must have: source1_entity_id , candidate_entity_ids (list)

    rows = []

    for row in test_candidates.itertuples(index=False):
        candidate_str = ",".join(row.candidate_entity_ids)
        rows.append(f"{row.source1_entity_id}\t{candidate_str}")

    with open(path,"w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        f.write("\n".join(rows))
        f.write("\n")

    print(f"Wrote {len(rows)} rows to {path}")


def write_matching_results(source1_df, predicted_features, path):
    """
    Writes matching_results.tsv: one row per Source1 test entity, with matched_entity_ids populated only from pairs where pred_label=1.

Every Source1 entity gets a row, including ones with zero matches(empty string) and ones with zero candidates at all (never appeared in predicted_features).
"""
    matches_by_source1 = {}
    
    for row in predicted_features.itertuples(index=False):
        if row.pred_label == 1:
            matches_by_source1.setdefault(
                row,source1_entity_id, []
            ).append(row.candidate_entity_id)

    rows = []

    for entity_id in source1_df["entity_id"]:
        matched_ids = matches_by_source1.get(entity_id,[])
        # De-duplicate defensively, preserve order
        seen = set()
        deduped = []
        for m in matched_ids:
            if m not in seen:
                seen.add(m)
                deduped.append(m)

        matched_str = ','.join(deduped)
        rows.append(f"{entity_id}\t{matched_str}")

    with open(path, "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("\n".join(rows))
        f.write("\n")

    print(f"Wrote {len(rows)} rows to {path}")