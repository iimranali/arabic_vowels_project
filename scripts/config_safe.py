from pathlib import Path


# ============================================================
# Main project paths
# ============================================================

PROJECT_ROOT = Path(r"C:\Workspace\arabic_vowels_project")

RAW_AUDIO_DIR = PROJECT_ROOT / "data" / "raw_audio"

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
AUG_AUDIO_DIR = PROJECT_ROOT / "data" / "augmented_audio"
FEATURES_DIR = PROJECT_ROOT / "data" / "features_mel_safe"

ORIGINAL_METADATA_CSV = METADATA_DIR / "metadata_original.csv"
SPLIT_METADATA_CSV = METADATA_DIR / "metadata_split_safe.csv"
FINAL_METADATA_CSV = METADATA_DIR / "metadata_features_safe.csv"


# ============================================================
# Audio / feature parameters
# ============================================================

SAMPLE_RATE = 16000
MAX_DURATION_SECONDS = 1.0

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256


# ============================================================
# Split settings
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================
# Augmentation settings
# ============================================================

AUGMENT_TRAIN_ONLY = True

# original + 4 augmented versions
AUGMENTATIONS_PER_TRAIN_SAMPLE = 4

NOISE_SNR_MIN = 18
NOISE_SNR_MAX = 30

TIME_SHIFT_MIN_MS = 40
TIME_SHIFT_MAX_MS = 100

TIME_STRETCH_MIN = 0.92
TIME_STRETCH_MAX = 1.08

PITCH_SHIFT_MIN = -1.0
PITCH_SHIFT_MAX = 1.0

GAIN_MIN_DB = -6
GAIN_MAX_DB = 6


# ============================================================
# Label settings
# ============================================================

NUM_LETTERS = 28
NUM_VOWELS = 3
NUM_PHONEMES = 84

NUM_DIRECT84 = 84

DIRECT84_LOSS_WEIGHT = 1.0
LETTER_LOSS_WEIGHT = 0.5
VOWEL_LOSS_WEIGHT = 0.25
MAKHRAJ_LOSS_WEIGHT = 0.25

VOWEL_NAMES = {
    0: "Fatha",
    1: "Kasra",
    2: "Damma",
}


# ============================================================
# Makhraj mapping
# IMPORTANT:
# Adjust this mapping according to your dataset letter order.
# Letter IDs must be 0 to 27.
# ============================================================

MAKHRAJ_NAMES = {
    0: "Al-Jawf",
    1: "Al-Halq",
    2: "Al-Lisan",
    3: "Al-Shafatan",
    4: "Al-Khayshum",
}

# Placeholder/default mapping.
# You MUST verify this according to your dataset letter order.
LETTER_TO_MAKHRAJ = {
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 2,
    7: 2,
    8: 2,
    9: 2,
    10: 2,
    11: 2,
    12: 2,
    13: 2,
    14: 2,
    15: 2,
    16: 2,
    17: 2,
    18: 2,
    19: 2,
    20: 2,
    21: 2,
    22: 2,
    23: 2,
    24: 3,
    25: 3,
    26: 3,
    27: 4,
}