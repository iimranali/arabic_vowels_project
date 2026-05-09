from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(r"C:\Workspace\arabic_vowels_project")

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

ORIGINAL_METADATA_CSV = METADATA_DIR / "metadata_original.csv"
SPLIT_METADATA_CSV = METADATA_DIR / "metadata_split_safe.csv"
FEATURES_METADATA_CSV = METADATA_DIR / "metadata_features_safe.csv"

NUM_CLASSES = 84
NUM_LETTERS = 28
NUM_VOWELS = 3
NUM_MAKHRAJ = 5


def assert_columns(df, required_cols, file_name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{file_name} missing columns: {missing}")


def validate_original_metadata():
    print("\n==============================")
    print("1. Validating metadata_original")
    print("==============================")

    df = pd.read_csv(ORIGINAL_METADATA_CSV)

    assert_columns(
        df,
        ["original_id", "audio_path", "class_id", "letter_id", "vowel_id", "makhraj_id"],
        "metadata_original.csv",
    )

    print("Rows:", len(df))
    print("Unique original_id:", df["original_id"].nunique())
    print("Unique class_id:", df["class_id"].nunique())
    print("Class min:", df["class_id"].min())
    print("Class max:", df["class_id"].max())
    print("Unique letter_id:", df["letter_id"].nunique())
    print("Unique vowel_id:", df["vowel_id"].nunique())
    print("Unique makhraj_id:", df["makhraj_id"].nunique())

    expected_classes = set(range(NUM_CLASSES))
    found_classes = set(df["class_id"].unique())

    missing_classes = sorted(expected_classes - found_classes)
    extra_classes = sorted(found_classes - expected_classes)

    if len(df) != 6229:
        raise RuntimeError(f"Expected 6229 original samples, found {len(df)}")

    if df["original_id"].nunique() != len(df):
        raise RuntimeError("Duplicate original_id found in metadata_original.csv")

    if missing_classes:
        raise RuntimeError(f"Missing class IDs: {missing_classes}")

    if extra_classes:
        raise RuntimeError(f"Unexpected class IDs: {extra_classes}")

    if df["class_id"].nunique() != NUM_CLASSES:
        raise RuntimeError(f"Expected {NUM_CLASSES} classes")

    if df["letter_id"].nunique() != NUM_LETTERS:
        raise RuntimeError(f"Expected {NUM_LETTERS} letters")

    if df["vowel_id"].nunique() != NUM_VOWELS:
        raise RuntimeError(f"Expected {NUM_VOWELS} vowels")

    if df["makhraj_id"].nunique() != NUM_MAKHRAJ:
        raise RuntimeError(f"Expected {NUM_MAKHRAJ} makhraj groups")

    # Validate derived labels
    bad_letter = df[df["letter_id"] != df["class_id"] // 3]
    bad_vowel = df[df["vowel_id"] != df["class_id"] % 3]

    if len(bad_letter) > 0:
        raise RuntimeError("letter_id != class_id // 3 for some rows")

    if len(bad_vowel) > 0:
        raise RuntimeError("vowel_id != class_id % 3 for some rows")

    print("\nClass distribution:")
    print(df["class_id"].value_counts().sort_index())

    print("\n✅ metadata_original.csv is valid.")


def validate_split_metadata():
    print("\n==============================")
    print("2. Validating metadata_split_safe")
    print("==============================")

    df = pd.read_csv(SPLIT_METADATA_CSV)

    assert_columns(
        df,
        ["original_id", "audio_path", "class_id", "letter_id", "vowel_id", "makhraj_id", "split"],
        "metadata_split_safe.csv",
    )

    print("Rows:", len(df))
    print("Unique original_id:", df["original_id"].nunique())
    print("\nSplit counts:")
    print(df["split"].value_counts())

    if len(df) != 6229:
        raise RuntimeError(f"Expected 6229 split rows, found {len(df)}")

    if df["original_id"].nunique() != len(df):
        raise RuntimeError("Duplicate original_id found in metadata_split_safe.csv")

    valid_splits = {"train", "val", "test"}
    found_splits = set(df["split"].unique())

    if found_splits != valid_splits:
        raise RuntimeError(f"Expected splits {valid_splits}, found {found_splits}")

    # Leakage check: one original_id must belong to one split only
    split_per_original = df.groupby("original_id")["split"].nunique()
    leaked = split_per_original[split_per_original > 1]

    if len(leaked) > 0:
        raise RuntimeError("Leakage detected: same original_id appears in multiple splits")

    # Class coverage per split
    print("\nUnique classes per split:")
    print(df.groupby("split")["class_id"].nunique())

    missing_by_split = {}
    expected_classes = set(range(NUM_CLASSES))

    for split_name, part in df.groupby("split"):
        found_classes = set(part["class_id"].unique())
        missing = sorted(expected_classes - found_classes)
        if missing:
            missing_by_split[split_name] = missing

    if missing_by_split:
        raise RuntimeError(f"Some splits are missing classes: {missing_by_split}")

    print("\nClass distribution by split:")
    print(pd.crosstab(df["class_id"], df["split"]))

    print("\n✅ metadata_split_safe.csv is valid.")


def validate_features_metadata():
    print("\n==============================")
    print("3. Validating metadata_features_safe")
    print("==============================")

    df = pd.read_csv(FEATURES_METADATA_CSV)

    assert_columns(
        df,
        [
            "original_id",
            "audio_path",
            "class_id",
            "letter_id",
            "vowel_id",
            "makhraj_id",
            "split",
            "variant",
            "is_augmented",
            "processed_audio_path",
            "feature_path",
        ],
        "metadata_features_safe.csv",
    )

    print("Rows:", len(df))
    print("Unique original_id:", df["original_id"].nunique())

    print("\nRows by split:")
    print(df["split"].value_counts())

    print("\nAugmentation by split:")
    print(pd.crosstab(df["split"], df["is_augmented"]))

    # Validation/test must have no augmentation
    bad_eval_aug = df[
        (df["split"].isin(["val", "test"])) &
        (df["is_augmented"] == 1)
    ]

    if len(bad_eval_aug) > 0:
        raise RuntimeError("Validation/test contain augmented samples. Not allowed.")

    # Train should have original + 4 augmented rows per original_id
    train_df = df[df["split"] == "train"]
    train_counts = train_df.groupby("original_id").size()

    bad_train_counts = train_counts[train_counts != 5]

    if len(bad_train_counts) > 0:
        raise RuntimeError(
            "Some train original_id values do not have exactly 5 rows "
            "(1 original + 4 augmented)."
        )

    # Val/test should have exactly 1 row per original_id
    for split_name in ["val", "test"]:
        part = df[df["split"] == split_name]
        counts = part.groupby("original_id").size()
        bad_counts = counts[counts != 1]

        if len(bad_counts) > 0:
            raise RuntimeError(
                f"Some {split_name} original_id values do not have exactly 1 row."
            )

    # Leakage check: each original_id belongs to one split
    split_per_original = df.groupby("original_id")["split"].nunique()
    leaked = split_per_original[split_per_original > 1]

    if len(leaked) > 0:
        raise RuntimeError("Leakage detected in feature metadata.")

    # Validate feature paths exist
    missing_features = []

    for path in df["feature_path"]:
        if not Path(path).exists():
            missing_features.append(path)

    if missing_features:
        print("\nMissing feature path examples:")
        for p in missing_features[:10]:
            print(p)
        raise RuntimeError(f"Missing feature files: {len(missing_features)}")

    print("\nVariant distribution:")
    print(df["variant"].value_counts())

    print("\nClass coverage by split:")
    print(df.groupby("split")["class_id"].nunique())

    print("\n✅ metadata_features_safe.csv is valid.")


def main():
    validate_original_metadata()
    validate_split_metadata()
    validate_features_metadata()

    print("\n====================================")
    print("✅ ALL SAFE METADATA VALIDATIONS PASS")
    print("====================================")


if __name__ == "__main__":
    main()