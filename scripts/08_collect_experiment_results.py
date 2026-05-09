import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"C:\Workspace\arabic_vowels_project")
RUNS_DIR = PROJECT_ROOT / "runs_torch"
OUTPUT_DIR = PROJECT_ROOT / "paper_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


EXPERIMENTS = [
    {
        "name": "Single-task ResNet18, direct 84-class",
        "run_dir": "safe_single_task_resnet18_84",
        "acc_key": "final_test_acc",
        "f1_key": "final_test_macro_f1",
        "metric_type": "direct84",
    },
    {
        "name": "Single-task ResNet18, vowel-only",
        "run_dir": "safe_single_task_resnet18_vowel",
        "acc_key": "final_test_acc",
        "f1_key": "final_test_macro_f1",
        "metric_type": "vowel",
    },
    {
        "name": "MTL ResNet18, Letter + Vowel, derived 84",
        "run_dir": "safe_mtl_resnet18_letter_vowel",
        "acc_key": "final_test_derived84_acc",
        "f1_key": "final_test_derived84_macro_f1",
        "metric_type": "derived84",
    },
    {
        "name": "MTL ResNet18, Letter + Vowel + Makhraj, derived 84",
        "run_dir": "safe_mtl_resnet18_letter_vowel_makhraj",
        "acc_key": "final_test_derived84_acc",
        "f1_key": "final_test_derived84_macro_f1",
        "metric_type": "derived84",
    },
    {
        "name": "MTL ResNet18, Direct84 + Letter + Vowel",
        "run_dir": "safe_mtl_resnet18_direct84_letter_vowel",
        "acc_key": "final_test_direct84_acc",
        "f1_key": "final_test_direct84_macro_f1",
        "metric_type": "direct84",
    },
    {
        "name": "MTL ResNet18, Direct84 + Letter + Vowel + Makhraj",
        "run_dir": "safe_mtl_resnet18_direct84_letter_vowel_makhraj",
        "acc_key": "final_test_direct84_acc",
        "f1_key": "final_test_direct84_macro_f1",
        "metric_type": "direct84",
    },
    {
        "name": "MTL EfficientNet-B0, Direct84 + Letter + Vowel + Makhraj",
        "run_dir": "safe_mtl_efficientnetb0_direct84_letter_vowel_makhraj",
        "acc_key": "final_test_direct84_acc",
        "f1_key": "final_test_direct84_macro_f1",
        "metric_type": "direct84",
    },
]


def load_summary(run_dir_name):
    path = RUNS_DIR / run_dir_name / "summary.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    rows = []

    for exp in EXPERIMENTS:
        summary = load_summary(exp["run_dir"])

        acc = summary.get(exp["acc_key"])
        f1 = summary.get(exp["f1_key"])

        if acc is None:
            raise KeyError(f"Missing {exp['acc_key']} in {exp['run_dir']}")

        if f1 is None:
            raise KeyError(f"Missing {exp['f1_key']} in {exp['run_dir']}")

        rows.append({
            "model": exp["name"],
            "metric_type": exp["metric_type"],
            "test_accuracy": acc,
            "test_macro_f1": f1,
            "test_accuracy_percent": round(acc * 100, 2),
            "test_macro_f1_percent": round(f1 * 100, 2),
            "run_dir": exp["run_dir"],
        })

    df = pd.DataFrame(rows)

    csv_path = OUTPUT_DIR / "final_results_table.csv"
    df.to_csv(csv_path, index=False)

    latex_rows = []

    for _, row in df.iterrows():
        model_name = row["model"]

        if model_name == "MTL ResNet18, Direct84 + Letter + Vowel + Makhraj":
            model_name = r"\textbf{" + model_name + "}"
            acc = r"\textbf{" + f"{row['test_accuracy_percent']:.2f}" + "}"
            f1 = r"\textbf{" + f"{row['test_macro_f1_percent']:.2f}" + "}"
        else:
            acc = f"{row['test_accuracy_percent']:.2f}"
            f1 = f"{row['test_macro_f1_percent']:.2f}"

        latex_rows.append(f"{model_name} & {acc} & {f1} \\\\")

    latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Leakage-Safe Test Results Across Model Variants}
\label{tab:leakage_safe_results}
\scriptsize
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabular}{p{0.53\columnwidth}cc}
\toprule
\textbf{Model} & \textbf{Acc. (\%)} & \textbf{Macro-F1 (\%)} \\
\midrule
""" + "\n".join(latex_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

    latex_path = OUTPUT_DIR / "final_results_table_latex.txt"

    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

    print("Saved:")
    print(csv_path)
    print(latex_path)

    print("\nFinal results:")
    print(df[["model", "test_accuracy_percent", "test_macro_f1_percent"]])


if __name__ == "__main__":
    main()