import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score

from features import FEATURE_COLUMNS


# ---------------------------------------------------------
# 7. Label training pairs using ground truth
# ---------------------------------------------------------

def build_ground_truth_lookup(ground_truth):
    """
    source1_entity_id -> set of true matched entity_ids
    """

    lookup = {}

    for row in ground_truth.itertuples(index=False):
        source1_id = row.source1_entity_id
        matched_ids = row.matched_entity_ids

        if isinstance(matched_ids, float) or not str(matched_ids).strip():
            lookup[source1_id] = set()
        else:
            lookup[source1_id] = {
                x.strip() for x in str(matched_ids).split(",") if x.strip()
            }

    return lookup


def label_pairs(features_df, ground_truth):
    """
    Adds a `label` column (1 = true match, 0 = not a match)
    to features_df based on train_ground_truth.tsv.
    """

    gt_lookup = build_ground_truth_lookup(ground_truth)

    features_df = features_df.copy()

    features_df["label"] = features_df.apply(
        lambda row: int(
            row["candidate_entity_id"]
            in gt_lookup.get(row["source1_entity_id"], set())
        ),
        axis=1,
    )

    return features_df


# ---------------------------------------------------------
# 8. Train the classifier
# ---------------------------------------------------------

def train_classifier(train_features, test_size=0.2, random_state=42):
    """
    Splits labeled train_features into train/val, trains an
    XGBoost classifier, and returns:

        model, X_val, y_val
    """

    X = train_features[FEATURE_COLUMNS]
    y = train_features["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "max_depth": 5,
        "eta": 0.2,
        "gamma": 4,
        "min_child_weight": 6,
        "subsample": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    }

    model = xgb.train(
        params, dtrain, num_boost_round=100,
        evals=[(dtrain, "train"), (dval, "validation")],
        verbose_eval=10,
    )

    return model, X_val, y_val


# ---------------------------------------------------------
# 9. Tune decision threshold toward precision (F_0.5)
# ---------------------------------------------------------

def tune_threshold(model, X_val, y_val):
    """
    Sweeps thresholds and picks the one maximizing F_0.5
    on the validation split.

    Returns: best_threshold, best_f_score
    """

    dval = xgb.DMatrix(X_val)
    probs = model.predict(dval)

    best_thresh, best_f = 0.5, 0.0

    for t in np.arange(0.3, 0.95, 0.05):
        preds = (probs > t).astype(int)
        f = fbeta_score(y_val, preds, beta=0.5)
        if f > best_f:
            best_f, best_thresh = f, t

    print(f"Best threshold: {best_thresh:.2f}  |  Val F_0.5: {best_f:.4f}")

    return best_thresh, best_f


# ---------------------------------------------------------
# 10. Save model + threshold
# ---------------------------------------------------------

def save_model(model, threshold, path="models/model.json"):
    model.save_model(path)

    threshold_path = str(path).replace(".json", "_threshold.txt")
    with open(threshold_path, "w") as f:
        f.write(str(threshold))

    print(f"Model saved to {path}")
    print(f"Threshold saved to {threshold_path}")