from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ============================================================
# DATA
# ============================================================

DATA_ROOT = (
    PROJECT_ROOT.parent
    / "dataset"
    / "6ab10eb3b23ba_student_resource"
    / "student_resource"
    / "dataset"
)

TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"


# ============================================================
# OUTPUT / MODELS
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TRAIN FILES
# ============================================================

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"


# ============================================================
# TEST FILES
# ============================================================

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"


# ============================================================
# MODEL SETTINGS
# ============================================================

MAX_TEXT_LENGTH = 256
EMBEDDING_DIM = 256
TOP_K = 50
BATCH_SIZE = 128
EPOCHS_ENCODER = 10
EPOCHS_MATCHER = 10
LEARNING_RATE = 1e-3

# ============================================================
# V1 TRAINING SETTINGS
# ============================================================

POSITIVE_PAIRS_V1 = 200_000
NEGATIVE_PAIRS_V1 = 200_000

ENCODER_BATCH_SIZE = 256
ENCODER_EPOCHS = 3
ENCODER_LEARNING_RATE = 1e-3
ENCODER_MARGIN = 0.5

NUM_WORKERS = 0

RETRIEVAL_TOP_K = 100
FINAL_MATCH_TOP_K = 20