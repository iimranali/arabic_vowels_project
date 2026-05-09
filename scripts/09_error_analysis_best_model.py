from pathlib import Path

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(r"C:\Workspace\arabic_vowels_project")

RUN_DIR = PROJECT_ROOT / "runs_torch" / "safe_mtl_resnet18_direct84_letter_vowel_makhraj"
REPORT_DIR = RUN_DIR / "reports"

OUTPUT_DIR = PROJECT_ROOT / "paper_outputs" / "error_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES_84 = 84
NUM_LETTERS = 28
NUM_VOWELS = 3

VOWEL_NAMES = {
    0: "Fatha",
    1: "Kasra",
    2: "Damma",
}


def class_to_letter_vowel(class_id):
    letter_id = class_id // 3
    vowel_id = class_id % 3
    return letter_id, vowel_id


def load_confusion_matrix(task_name):
    path = REPORT_DIR / f"test_{task_name}_confusion_matrix.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing confusion matrix: {path}")

    df = pd.read_csv(path)

    # Some saved matrices may have an unnamed index column
    if df.columns[0].lower().startswith("unnamed"):
        df = df.drop(columns=[df.columns[0]])

    return df.values


def top_confusions_from_cm(cm, top_k=30):
    rows = []

    for true_id in range(cm.shape[0]):
        for pred_id in range(cm.shape[1]):
            count = int(cm[true_id, pred_id])

            if true_id == pred_id:
                continue

            if count <= 0:
                continue

            true_letter, true_vowel = class_to_letter_vowel(true_id)
            pred_letter, pred_vowel = class_to_letter_vowel(pred_id)

            rows.append({
                "true_class": true_id,
                "pred_class": pred_id,
                "count": count,

                "true_letter": true_letter,
                "pred_letter": pred_letter,

                "true_vowel": VOWEL_NAMES[true_vowel],
                "pred_vowel": VOWEL_NAMES[pred_vowel],

                "same_letter": true_letter == pred_letter,
                "same_vowel": true_vowel == pred_vowel,
            })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return df

    df = df.sort_values("count", ascending=False).head(top_k)

    return df


def summarize_errors(cm):
    total = cm.sum()
    correct = np.trace(cm)
    errors = total - correct

    same_vowel_errors = 0
    same_letter_errors = 0
    different_letter_same_vowel = 0
    same_letter_different_vowel = 0
    different_letter_different_vowel = 0

    for true_id in range(cm.shape[0]):
        for pred_id in range(cm.shape[1]):
            if true_id == pred_id:
                continue

            count = int(cm[true_id, pred_id])

            if count == 0:
                continue

            true_letter, true_vowel = class_to_letter_vowel(true_id)
            pred_letter, pred_vowel = class_to_letter_vowel(pred_id)

            same_letter = true_letter == pred_letter
            same_vowel = true_vowel == pred_vowel

            if same_vowel:
                same_vowel_errors += count

            if same_letter:
                same_letter_errors += count

            if (not same_letter) and same_vowel:
                different_letter_same_vowel += count

            elif same_letter and (not same_vowel):
                same_letter_different_vowel += count

            else:
                different_letter_different_vowel += count

    summary = {
        "total_samples": int(total),
        "correct": int(correct),
        "errors": int(errors),
        "accuracy": float(correct / total),

        "same_vowel_errors": int(same_vowel_errors),
        "same_letter_errors": int(same_letter_errors),

        "different_letter_same_vowel": int(different_letter_same_vowel),
        "same_letter_different_vowel": int(same_letter_different_vowel),
        "different_letter_different_vowel": int(different_letter_different_vowel),

        "same_vowel_error_percent": float(100 * same_vowel_errors / max(errors, 1)),
        "same_letter_error_percent": float(100 * same_letter_errors / max(errors, 1)),
    }

    return summary


def main():
    direct84_cm = load_confusion_matrix("direct84_84")
    derived84_cm = load_confusion_matrix("derived84_84")

    direct_top = top_confusions_from_cm(direct84_cm, top_k=30)
    derived_top = top_confusions_from_cm(derived84_cm, top_k=30)

    direct_top_path = OUTPUT_DIR / "top_confusions_direct84.csv"
    derived_top_path = OUTPUT_DIR / "top_confusions_derived84.csv"

    direct_top.to_csv(direct_top_path, index=False)
    derived_top.to_csv(derived_top_path, index=False)

    direct_summary = summarize_errors(direct84_cm)
    derived_summary = summarize_errors(derived84_cm)

    summary_df = pd.DataFrame([
        {"task": "direct84", **direct_summary},
        {"task": "derived84", **derived_summary},
    ])

    summary_path = OUTPUT_DIR / "error_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("Saved:")
    print(direct_top_path)
    print(derived_top_path)
    print(summary_path)

    print("\nError summary:")
    print(summary_df)

    print("\nTop direct84 confusions:")
    print(direct_top.head(10))


if __name__ == "__main__":
    main()