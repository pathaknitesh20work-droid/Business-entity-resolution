# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** CodeBlooded  

**Team Members:** Anwesha Rudra
                  Nitesh Patak
                  Aarav Raj  
                  Chikkala Pardha Saradhi

**Submission Date:** 27 September 2026

---

## 1. Executive Summary

Our solution uses a **hybrid blocking + supervised classification pipeline** for business entity resolution across Source 1, Source 2, and Source 3. We first normalize business names, addresses, and country values, then retrieve the top candidate entities from the combined Source 2/Source 3 pool using **character n-gram TF-IDF and cosine nearest-neighbor search**. Each candidate pair is scored with semantic embedding similarities, fuzzy string similarities, and country agreement features, and an **XGBoost binary classifier** makes the final match decision using a validation-tuned precision-oriented threshold.

The main design goal is to reduce the otherwise very large cross-source comparison space while retaining strong matching signals. The final submitted test run contains **17,326 Source 1 entities**, **173,260 blocking candidate relationships** (10 candidates per Source 1 entity), and **3,185 predicted matches**.

---

## 2. Methodology

### 2.1 Problem Analysis

Business entity resolution is difficult because the same real-world organization can appear differently across data sources. The implemented pipeline is designed around the following data-quality and matching issues:

- **Business-name variation:** case differences, punctuation, spacing, and corporate suffixes such as `Ltd`, `Limited`, `LLC`, `Inc`, `Corp`, `Pvt Ltd`, and similar variants can make exact string matching unreliable.
- **Address variation:** address abbreviations such as `rd`, `st`, `ave`, `blvd`, `apt`, `fl`, and `no` can cause the same location to be represented differently.
- **Country aliases:** equivalent country representations such as `USA`, `United States`, `United States of America`, `IN`, and `FR` require standardization.
- **Approximate text similarity:** typographical differences, word reordering, partial names, and differently formatted addresses require both fuzzy and semantic similarity signals.
- **Large comparison space:** comparing every Source 1 record with every Source 2 and Source 3 record would be computationally expensive, so blocking is required before detailed pairwise scoring.
- **Singletons / zero-match entities:** the pipeline preserves one output row for every Source 1 entity, including entities for which no Source 2/Source 3 match passes the classifier threshold.

The project also includes a stratified sampling utility for local development. For training subsets, it preserves country distribution, the Source 1 singleton-versus-matched distribution, and the referential integrity of matched Source 2/Source 3 IDs. Test sampling is country-stratified only because test ground truth is unavailable.

### 2.2 Solution Strategy

The end-to-end pipeline is:

1. Load training and test TSV files.
2. Normalize business names, addresses, and country values using the same preprocessing rules for all three sources.
3. Concatenate normalized Source 2 and Source 3 into a single candidate pool while preserving their original entity IDs.
4. Build a character n-gram TF-IDF blocking index over normalized candidate business names.
5. Retrieve the nearest candidate entities for each Source 1 entity using cosine-distance nearest neighbors.
6. Flatten the candidate lists into individual Source 1–candidate pairs.
7. Build pairwise similarity features using sentence embeddings, fuzzy string comparisons, and country equality.
8. Label training candidate pairs from `train_ground_truth.tsv`.
9. Train an XGBoost binary classifier on an 80/20 stratified train/validation split.
10. Sweep decision thresholds from 0.30 to 0.90 in steps of 0.05 and select the threshold that maximizes validation F0.5.
11. Apply the trained model and selected threshold to test candidate pairs.
12. Write the blocking candidates to `output/candidate_pairs.tsv` and final accepted matches to `output/matching_results.tsv`.

**Approach Type:** Hybrid — TF-IDF/Nearest-Neighbor Blocking + Feature-Based XGBoost Classifier  
**Core Innovation:** A two-stage retrieval-and-classification design combining character-level lexical retrieval with both semantic embedding similarity and fuzzy string similarity. The feature stage embeds only entities that actually occur in the blocked candidate pairs, avoiding unnecessary embedding computation over the complete source pools.

---

## 3. Candidate Generation (Blocking)

The blocking stage reduces the search space before expensive feature construction and model inference.

A single candidate pool is created by concatenating normalized Source 2 and Source 3 records. A `TfidfVectorizer` is fitted on each candidate's normalized business name with:

- analyzer: `char_wb`
- character n-gram range: **2 to 4**
- `min_df = 1`

The resulting sparse TF-IDF matrix is indexed with `sklearn.neighbors.NearestNeighbors` using **cosine distance**. Each normalized Source 1 business name is transformed using the same vectorizer and queried against the candidate index.

- **Blocking keys used:** normalized business name represented by character-level TF-IDF 2–4 grams, followed by cosine nearest-neighbor retrieval.
- **Configured retrieval depth for the supplied run:** `RETRIEVAL_TOP_K = 10`.
- **Source 1 entities in submitted candidate file:** **17,326**.
- **Candidates per Source 1 entity:** **10** for every submitted Source 1 row.
- **Total candidate relationships represented:** **173,260**.
- **Candidate file format:** one row per Source 1 entity, with the 10 retrieved Source 2/Source 3 IDs stored as a comma-separated list in `candidate_entity_ids`.
- **How true matches are protected:** the blocker uses character n-grams rather than exact name equality, so minor punctuation, spelling, spacing, and substring-level differences can still retrieve similar business names. The subsequent classifier uses address and semantic features to distinguish the retrieved alternatives. However, no empirical blocking-recall figure is stored in the supplied artifacts, so candidate recall is not claimed here.

For development sampling, the supplied stratified sampling script explicitly retains every ground-truth Source 2/Source 3 ID referenced by a sampled Source 1 training entity, preserving referential integrity in sampled training data.

---

## 4. Matching Model

After blocking, each Source 1–candidate pair is represented by six features.

**Features used:**

- **Name features:**
  - `name_cos`: cosine similarity between normalized business-name embeddings.
  - `name_fuzz`: RapidFuzz token-sort similarity between normalized names.
  - `name_partial`: RapidFuzz partial-ratio similarity between normalized names.
- **Address features:**
  - `addr_cos`: cosine similarity between normalized business-address embeddings.
  - `addr_fuzz`: RapidFuzz token-sort similarity between normalized addresses.
- **Other:**
  - `country_match`: binary indicator equal to 1 when normalized country values are identical and 0 otherwise.

### Text Embeddings

The supplied implementation loads `sentence-transformers/all-MiniLM-L6-v2` by default. Embeddings are L2-normalized by SentenceTransformers, so the dot product used in the feature code is equivalent to cosine similarity. The code comments also note `Qwen/Qwen3-Embedding-0.6B` as a potential larger-model option for a higher-resource AWS run, but the default implementation in the submitted source uses MiniLM unless the model name is changed explicitly.

To reduce runtime and memory consumption, embeddings are built only for Source 1 and Source 2/3 entities that appear in the blocked pair set, rather than embedding the complete source tables.

### Classifier

**Model type:** XGBoost binary classifier (`binary:logistic`).

The supplied training parameters are:

| Parameter | Value |
|---|---:|
| `max_depth` | 5 |
| `eta` | 0.2 |
| `gamma` | 4 |
| `min_child_weight` | 6 |
| `subsample` | 0.8 |
| `objective` | `binary:logistic` |
| `eval_metric` | `logloss` |
| Boosting rounds | 100 |

Training candidate pairs are labeled by checking whether each `candidate_entity_id` appears in the comma-separated ground-truth matches for the corresponding `source1_entity_id`.

The labeled candidate-pair data is split into **80% training / 20% validation** with `random_state = 42`, using class-label stratification.

**Threshold selection method:** The validation probabilities are evaluated at thresholds from **0.30 through 0.90 in steps of 0.05**. The selected threshold is the one maximizing **F0.5**, which weights precision more strongly than recall. The supplied saved threshold is **0.75**.

At inference time, a candidate is accepted as a match only when its predicted probability is strictly greater than the selected threshold. Accepted matches are grouped by Source 1 entity and written to `matching_results.tsv`; Source 1 entities with no accepted matches are still written with an empty `matched_entity_ids` value.

---

## 5. Results & Error Analysis

### Output Statistics

The supplied final output files contain the following:

| Statistic | Value |
|---|---:|
| Source 1 rows in `candidate_pairs.tsv` | 17,326 |
| Candidates per Source 1 entity | 10 |
| Total candidate relationships | 173,260 |
| Source 1 rows in `matching_results.tsv` | 17,326 |
| Source 1 entities with at least one predicted match | 1,969 |
| Source 1 entities with zero predicted matches | 15,357 |
| Total predicted matched entity IDs | 3,185 |
| Predicted Source 2 matches | 1,510 |
| Predicted Source 3 matches | 1,675 |
| Maximum predicted matches for one Source 1 entity | 9 |
| Saved decision threshold | 0.75 |

- **F0.5 Score (macro):** The exact numeric validation F0.5 value is **not stored in the supplied model, output, or log artifacts**. The training code computes and prints the best validation F0.5 while tuning the threshold, but only the selected threshold (0.75) is persisted. Therefore, an exact score is not reported here rather than inventing a value.

### Error Analysis

The supplied artifacts do not include a saved validation-prediction table or false-positive/false-negative report, so the following are **code-informed risk areas rather than measured error counts**:

- **Potential false positives (wrong merges):** businesses with very similar or identical names can enter the same top-name candidate neighborhood, especially when address text is incomplete, generic, or noisy. Shared country values and high fuzzy name similarity can further strengthen these ambiguous pairs. The precision-oriented F0.5 threshold tuning and the relatively high saved threshold of 0.75 are intended to reduce this risk.
- **Potential false negatives (missed matches):** a true match cannot be recovered by the classifier if it is not present in the top-10 TF-IDF name candidates. Large name changes, aliases, severe spelling corruption, transliteration differences, or weak/empty normalized names can therefore create blocking misses. A second source of false negatives is a true candidate receiving probability at or below the strict 0.75 decision threshold.
- **Address sensitivity:** standardized abbreviations improve consistency, but semantically equivalent addresses with substantially different formatting or missing components can still reduce both embedding and fuzzy address similarity.
- **Country normalization coverage:** only explicitly coded country aliases are canonicalized, so unseen abbreviations or alternative spellings are left unchanged and may cause `country_match = 0` even for a valid cross-source pair.

A future evaluation run should persist validation probabilities, pair labels, blocking recall, false-positive examples, and false-negative examples so these error categories can be quantified directly.

---

## 6. Conclusion

The CodeBlooded solution implements a scalable two-stage business entity resolution pipeline that first narrows the comparison space with character n-gram TF-IDF nearest-neighbor blocking and then applies an XGBoost classifier using semantic, fuzzy-text, address, and country features. The submitted run reduces matching to 10 candidates per Source 1 entity and uses a validation-selected F0.5 decision threshold of 0.75, while preserving a complete output row for every Source 1 entity.

The primary lesson from the implementation is that entity resolution benefits from combining complementary signals: lexical blocking provides computational efficiency, while semantic embeddings and fuzzy comparisons improve robustness to noisy business names and addresses. The main remaining evaluation gap in the supplied artifacts is that the exact validation F0.5 score and empirical error cases were not persisted, even though threshold tuning is implemented in code.

---

## Appendix

### A. Code Artefacts

The solution code is organized under:

```text
code/business_entity_resolution/
├── src/
│   ├── run_pipeline.py
│   ├── config.py
│   ├── data_loader.py
│   ├── data_pipeline.py
│   ├── preprocessing.py
│   ├── normalization.py
│   ├── blocking.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   ├── analyze_ground_truth.py
│   └── stratified_sampling.py
├── models/
│   ├── model.json
│   └── model_threshold.txt
├── README.md
└── requirements.txt
```

#### Main Entry Point

The end-to-end entry point is:

```bash
cd code/business_entity_resolution/src
python run_pipeline.py
```

`run_pipeline.py` performs the following operations in sequence:

```text
Load train/test TSVs
        ↓
Normalize names, addresses, countries
        ↓
Combine Source 2 + Source 3
        ↓
TF-IDF / cosine NearestNeighbors blocking
        ↓
Flatten Source 1–candidate pairs
        ↓
Build embedding + fuzzy + country features
        ↓
Label training pairs from ground truth
        ↓
Train XGBoost classifier
        ↓
Tune F0.5 decision threshold
        ↓
Save/reload model + threshold
        ↓
Predict test candidate pairs
        ↓
Write candidate_pairs.tsv
        ↓
Write matching_results.tsv
```

#### Important Source Files

- `normalization.py` — lowercases text, removes punctuation, collapses whitespace, removes common company legal suffixes, expands selected address abbreviations, and standardizes selected country aliases.
- `preprocessing.py` — applies normalization consistently to Source 1, Source 2, and Source 3 and concatenates Source 2/3 for retrieval.
- `blocking.py` — builds the character n-gram TF-IDF + cosine nearest-neighbor index and produces candidate IDs.
- `features.py` — generates MiniLM name/address embeddings and RapidFuzz similarities plus country agreement.
- `train.py` — labels pairs, trains XGBoost, tunes F0.5 threshold, and saves the model/threshold.
- `predict.py` — loads the trained model, applies the threshold, and writes the two required output TSVs.
- `stratified_sampling.py` — creates smaller development subsets while preserving training country/match strata and training-match referential integrity.
- `analyze_ground_truth.py` — provides utilities for analyzing match-count and Source 2/Source 3 distributions in the training ground truth.

#### Expected Data Layout

The configured loader expects the dataset to contain:

```text
data/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

The exact parent path is controlled by `DATA_ROOT` in `src/config.py`. Before reproducing on another machine, this path should be checked so it points to the intended dataset location.

#### Required Output Files

The pipeline produces:

```text
output/
├── candidate_pairs.tsv
└── matching_results.tsv
```

`candidate_pairs.tsv` contains:

```text
source1_entity_id    candidate_entity_ids
```

`matching_results.tsv` contains:

```text
source1_entity_id    matched_entity_ids
```

### B. Additional Results

#### Candidate Set Distribution

For the supplied output, every Source 1 entity has exactly 10 blocking candidates:

- minimum candidates: 10
- median candidates: 10
- mean candidates: 10
- maximum candidates: 10

This is consistent with `RETRIEVAL_TOP_K = 10` in the supplied configuration and produces 173,260 candidate relationships from 17,326 Source 1 entities.

#### Final Prediction Distribution

From the supplied `matching_results.tsv`:

- **15,357** Source 1 entities have no predicted match.
- **1,969** Source 1 entities have one or more predicted matches.
- **3,185** matched entity IDs are predicted in total.
- **1,510** predicted IDs originate from Source 2.
- **1,675** predicted IDs originate from Source 3.
- The largest predicted match list for a single Source 1 entity contains **9 IDs**.

These figures describe the submitted prediction file only; they are not accuracy metrics because test ground truth was not supplied with the uploaded artifacts.

---

**Note:** The methodology above is based directly on the supplied CodeBlooded source code, saved model threshold, and final output TSV files. The exact validation F0.5 score and empirical false-positive/false-negative samples were not present in the supplied artifacts and have therefore not been fabricated.
