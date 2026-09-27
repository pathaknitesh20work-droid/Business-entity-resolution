from data_loader import load_tsv
from config import TRAIN_GROUND_TRUTH,TRAIN_SOURCE1,TRAIN_SOURCE2,TRAIN_SOURCE3,TEST_SOURCE1,TEST_SOURCE2,TEST_SOURCE3,RETRIEVAL_TOP_K,MODEL_DIR
from preprocessing import normalize_entity_data,combine_sources
from blocking import generate_candidates,flatten_candidates
from features import load_embedding_model, build_features, FEATURE_COLUMNS
from train import label_pairs, train_classifier, tune_threshold, save_model
from predict import load_model, predict_matches, write_candidate_pairs, write_matching_results
from config import OUTPUT_DIR, MODEL_DIR

if __name__ == "__main__":
    ## loading data
    train_ground_truth = load_tsv(TRAIN_GROUND_TRUTH)
    train_source1 = load_tsv(TRAIN_SOURCE1)
    train_source2 = load_tsv(TRAIN_SOURCE2)
    train_source3 = load_tsv(TRAIN_SOURCE3)
    test_source1, test_source2, test_source3 = load_tsv(TEST_SOURCE1),load_tsv(TEST_SOURCE2),load_tsv(TEST_SOURCE3)
    print("1 Data Loaded successfully")

    ## Normalizing data
    norm_train_source1, norm_train_source2, norm_train_source3 = normalize_entity_data(train_source1,train_source2,train_source3)
    norm_test_source1, norm_test_source2, norm_test_source3 = normalize_entity_data(test_source1,test_source2,test_source3)
    print("2 Data normalized successfully")

    ## combine source2 & source3 into one 
    train_s2s3 = combine_sources(norm_train_source2,norm_train_source3)
    test_s2s3 = combine_sources(norm_test_source2,norm_test_source3)
    print("3 Combined training/testing data source 2 & 3 successfully")

    ## Candidate generation
    train_candidates = generate_candidates(norm_train_source1, train_s2s3, k=RETRIEVAL_TOP_K)
    test_candidates = generate_candidates(norm_test_source1, test_s2s3, k=RETRIEVAL_TOP_K)  
    print("4 Candidates generated successfully")

    ## Flatten candidates
    train_pairs = flatten_candidates(train_candidates)
    test_pairs = flatten_candidates(test_candidates)
    print("5 Candidate pairs flattened successfully")

    #Load model
    embed_model = load_embedding_model()
    print("Model loaded successfully")

    #set x,y
    train_features = build_features(train_pairs, norm_train_source1, train_s2s3, embed_model)
    test_features = build_features(test_pairs, norm_test_source1, test_s2s3, embed_model)

    train_features = label_pairs(train_features, train_ground_truth)
    print("Labels and features defined successfully")

    #Train model
    model, X_val, y_val = train_classifier(train_features)
    best_threshold, best_f = tune_threshold(model, X_val, y_val)
    print("Model trained successfully")

    # Save the model
    save_model(model, best_threshold, path=MODEL_DIR / "model.json")
    print("Model saved successfully") 
    model, threshold = load_model(MODEL_DIR/ "model.json")

    #Make prediction
    test_predictions = predict_matches(model, test_features, threshold)
    print("Prediction have been made")

    write_candidate_pairs(test_candidates, OUTPUT_DIR / "candidate_pairs.tsv")
    print("candidate_pairs.tsv written successfully")
    write_matching_results(norm_test_source1, test_predictions, OUTPUT_DIR / "matching_results.tsv")
    print("matching_results.tsv written successfully")
    