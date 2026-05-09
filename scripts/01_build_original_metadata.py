import re
from pathlib import Path

import pandas as pd

from config_safe import (
    RAW_AUDIO_DIR,
    METADATA_DIR,
    ORIGINAL_METADATA_CSV,
    NUM_PHONEMES,
    NUM_VOWELS,
    LETTER_TO_MAKHRAJ,
)


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def extract_class_id(audio_path: Path) -> int:
    """
    Extract class id from parent folder or filename.

    This dataset is expected to use 1-based class folders:
        1, 2, 3, ..., 84

    We convert them to 0-based class IDs:
        0, 1, 2, ..., 83
    """

    candidates = [
        audio_path.parent.name,
        audio_path.stem,
    ]

    for text in candidates:
        numbers = re.findall(r"\d+", text)

        if numbers:
            raw_id = int(numbers[-1])

            # Dataset folders are expected to be 1-based: 1 to 84
            if 1 <= raw_id <= NUM_PHONEMES:
                return raw_id - 1

    raise ValueError(f"Could not extract class id from: {audio_path}")


def main():
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    audio_files = [
        p for p in RAW_AUDIO_DIR.rglob("*")
        if p.suffix.lower() in AUDIO_EXTENSIONS
    ]

    if not audio_files:
        raise FileNotFoundError(f"No audio files found in: {RAW_AUDIO_DIR}")

    rows = []

    for idx, audio_path in enumerate(sorted(audio_files)):
        class_id = extract_class_id(audio_path)

        letter_id = class_id // NUM_VOWELS
        vowel_id = class_id % NUM_VOWELS
        makhraj_id = LETTER_TO_MAKHRAJ[letter_id]

        rows.append({
            "original_id": f"orig_{idx:06d}",
            "audio_path": str(audio_path),
            "class_id": class_id,
            "letter_id": letter_id,
            "vowel_id": vowel_id,
            "makhraj_id": makhraj_id,
        })

    df = pd.DataFrame(rows)

    print("Original metadata summary")
    print("-------------------------")
    print("Total samples:", len(df))
    print("Unique 84 classes:", df["class_id"].nunique())
    print("Unique letters:", df["letter_id"].nunique())
    print("Unique vowels:", df["vowel_id"].nunique())
    print("Unique makhraj:", df["makhraj_id"].nunique())

    print("\nClass distribution:")
    print(df["class_id"].value_counts().sort_index())

    expected_classes = set(range(NUM_PHONEMES))
    found_classes = set(df["class_id"].unique())

    missing_classes = sorted(expected_classes - found_classes)
    extra_classes = sorted(found_classes - expected_classes)

    if missing_classes:
        raise RuntimeError(f"Missing class IDs: {missing_classes}")

    if extra_classes:
        raise RuntimeError(f"Unexpected class IDs: {extra_classes}")

    if df["class_id"].nunique() != NUM_PHONEMES:
        raise RuntimeError(
            f"Expected {NUM_PHONEMES} classes, found {df['class_id'].nunique()}"
        )

    df.to_csv(ORIGINAL_METADATA_CSV, index=False)

    print("\nSaved:")
    print(ORIGINAL_METADATA_CSV)


if __name__ == "__main__":
    main()