from pathlib import Path

ROOT = Path("C:\Users\91628\OneDrive\Desktop\amazon_ml_challenge\dataset\6ab10eb3b23ba_student_resource\student_resource\dataset").resolve().parent.parent

TRAIN_DIR = "C:\Users\91628\OneDrive\Desktop\amazon_ml_challenge\dataset\6ab10eb3b23ba_student_resource.zip\student_resource\dataset\train"
TEST_DIR = "C:\Users\91628\OneDrive\Desktop\amazon_ml_challenge\dataset\6ab10eb3b23ba_student_resource.zip\student_resource\dataset\test"

MODEL_DIR = ROOT / "models"
OUTPUT_DIR = "C:\Users\91628\OneDrive\Desktop\amazon_ml_challenge\CodeBlooded_submission\output"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_SOURCE1 = "C:\Users\91628\OneDrive\Desktop\amazon_ml_challenge\dataset\6ab10eb3b23ba_student_resource.zip\student_resource\dataset\train\train_source1.tsv"
TRAIN_SOURCE2 =  "C:\Users\91628\OneDrive\Desktop\amazon_ml_challenge\dataset\6ab10eb3b23ba_student_resource.zip\student_resource\dataset\train\train_source2.tsv"
TRAIN_SOURCE3 =  "C:\Users\91628\OneDrive\Desktop\amazon_ml_challenge\dataset\6ab10eb3b23ba_student_resource.zip\student_resource\dataset\train\train_source3.tsv"
TRAIN_GROUND_TRUTH =  "C:\Users\91628\OneDrive\Desktop\amazon_ml_challenge\dataset\6ab10eb3b23ba_student_resource.zip\student_resource\dataset\train\train_ground_truth.tsv"

TEST_SOURCE1 =  "C:\Users\91628\OneDrive\Desktop\amazon_ml_challenge\dataset\6ab10eb3b23ba_student_resource.zip\student_resource\dataset\test\test_source1.tsv"
TEST_SOURCE2 =  "C:\Users\91628\OneDrive\Desktop\amazon_ml_challenge\dataset\6ab10eb3b23ba_student_resource.zip\student_resource\dataset\test\test_source2.tsv"
TEST_SOURCE3 =  "C:\Users\91628\OneDrive\Desktop\amazon_ml_challenge\dataset\6ab10eb3b23ba_student_resource.zip\student_resource\dataset\test\test_source3.tsv"

MAX_TEXT_LENGTH = 256

EMBEDDING_DIM = 256

TOP_K = 50

BATCH_SIZE = 128

EPOCHS_ENCODER = 10
EPOCHS_MATCHER = 10

LEARNING_RATE = 1e-3