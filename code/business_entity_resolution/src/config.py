from pathlib import Path
## NOTE : THIS FILE HAS PATHS RELATIVE TO SAMEER'S LOCAL MACHINE, DO NOT USE THIS VERSION OF CONFIG.PY FOR AWS INSTANCE 

# src/ -> business_entity_resolution/ -> code/ -> CodeBlooded_submission/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_ROOT = PROJECT_ROOT.parent / "data" / "dataset_sample"   # <- using sample locally
# DATA_ROOT = PROJECT_ROOT.parent / "data" / "dataset"        # <- switch back for full run

TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"

OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

RETRIEVAL_TOP_K = 10 # Set to 100 for the full real run
FINAL_MATCH_TOP_K = 20