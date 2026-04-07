from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT
DATA_DIR = ROOT / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
PROCESSED_FRAMES_DIR = DATA_DIR / "processed_frames"
METADATA_DIR = DATA_DIR / "metadata"
DEFAULT_DATASET_CSV = METADATA_DIR / "dataset.csv"

SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
OUTPUTS_DIR = ROOT / "outputs"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
INDEX_DIR = OUTPUTS_DIR / "index"
RESULTS_DIR = OUTPUTS_DIR / "results"
