import pandas as pd
from sklearn.model_selection import train_test_split

from config_safe import (
    ORIGINAL_METADATA_CSV,
    SPLIT_METADATA_CSV,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)


def main():
    df = pd.read_csv(ORIGINAL_METADATA_CSV)

    if "original_id" not in df.columns:
        raise ValueError("metadata_original.csv must contain original_id column.")

    if "class_id" not in df.columns:
        raise ValueError("metadata_original.csv must contain class_id column.")

    if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-6:
        raise ValueError("Train/val/test ratios must sum to 1.")

    # First split: train vs temp
    train_df, temp_df = train_test_split(
        df,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_SEED,
        stratify=df["class_id"],
        shuffle=True,
    )

    # Second split: validation vs test
    val_fraction_inside_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_fraction_inside_temp,
        random_state=RANDOM_SEED,
        stratify=temp_df["class_id"],
        shuffle=True,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Leakage check: every original_id must appear in only one split
    leakage_check = final_df.groupby("original_id")["split"].nunique()
    leaked = leakage_check[leakage_check > 1]

    if len(leaked) > 0:
        raise RuntimeError("Leakage detected: same original_id appears in multiple splits.")

    print("Leakage-safe split summary")
    print("--------------------------")
    print(final_df["split"].value_counts())

    print("\nClass count per split:")
    print(pd.crosstab(final_df["class_id"], final_df["split"]))

    final_df.to_csv(SPLIT_METADATA_CSV, index=False)

    print("\nSaved:")
    print(SPLIT_METADATA_CSV)


if __name__ == "__main__":
    main()