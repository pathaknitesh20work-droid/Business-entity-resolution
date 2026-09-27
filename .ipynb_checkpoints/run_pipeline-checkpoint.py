from data_loader import load_tsv
from config import TRAIN_GROUND_TRUTH,TRAIN_SOURCE1,TRAIN_SOURCE2,TRAIN_SOURCE3,TEST_SOURCE1,TEST_SOURCE2,TEST_SOURCE3,RETRIEVAL_TOP_K,MODEL_DIR
from preprocessing import normalize_entity_data,combine_sources
from blocking import generate_candidates,flatten_candidates
from features import load_embedding_model, build_features, FEATURE_COLUMNS
from train import label_pairs, train_classifier, tune_threshold, save_model


if __name__ == "__main__":
    ## loading data
    train_ground_truth = load_tsv(TRAIN_GROUND_TRUTH)
    train_source1 = load_tsv(TRAIN_SOURCE1)
    train_source2 = load_tsv(TRAIN_SOURCE2)
    train_source3 = load_tsv(TRAIN_SOURCE3)
    test_source1, test_source2, test_source3 = load_tsv(TEST_SOURCE1),load_tsv(TEST_SOURCE2),load_tsv(TEST_SOURCE3)

    ## Normalizing data
    norm_train_source1, norm_train_source2, norm_train_source3 = normalize_entity_data(train_source1,train_source2,train_source3)
    norm_test_source1, norm_test_source2, norm_test_source3 = normalize_entity_data(test_source1,test_source2,test_source3)

    ## combine source2 & source3 into one 
    train_s2s3 = combine_sources(norm_train_source2,norm_train_source3)
    test_s2s3 = combine_sources(norm_test_source2,norm_test_source3)

    train_candidates = generate_candidates(norm_train_source1, train_s2s3, k=RETRIEVAL_TOP_K)
    test_candidates = generate_candidates(norm_test_source1, test_s2s3, k=RETRIEVAL_TOP_K)  

    train_pairs = flatten_candidates(train_candidates)
    test_pairs = flatten_candidates(test_candidates)

    embed_model = load_embedding_model()
    train_features = build_features(train_pairs, norm_train_source1, train_s2s3, embed_model)
    test_features = build_features(test_pairs, norm_test_source1, test_s2s3, embed_model)

    train_features = label_pairs(train_features, train_ground_truth)

    model, X_val, y_val = train_classifier(train_features)
    best_threshold, best_f = tune_threshold(model, X_val, y_val)

    save_model(model, best_threshold, path=MODEL_DIR / "model.json")