import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18


# ============================================================
# Config
# ============================================================

PROJECT_ROOT = Path(r"C:\Workspace\arabic_vowels_project")

METADATA_CSV = PROJECT_ROOT / "data" / "metadata" / "metadata_features_safe.csv"

RUN_DIR = PROJECT_ROOT / "runs_torch" / "safe_mtl_resnet18_direct84_aux_light"
REPORT_DIR = RUN_DIR / "reports"

NUM_DIRECT84 = 84
NUM_LETTERS = 28
NUM_VOWELS = 3
NUM_MAKHRAJ = 5

# Main task should dominate.
DIRECT84_LOSS_WEIGHT = 1.0
LETTER_LOSS_WEIGHT = 0.3
VOWEL_LOSS_WEIGHT = 0.1
MAKHRAJ_LOSS_WEIGHT = 0.2

SEED = 42

BATCH_SIZE = 64
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 4

USE_SPEC_AUGMENT = True
TIME_MASK = 20
FREQ_MASK = 16
NUM_MASKS = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAKHRAJ_NAMES = [
    "Al-Jawf",
    "Al-Halq",
    "Al-Lisan",
    "Al-Shafatan",
    "Al-Khayshum",
]


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# ============================================================
# SpecAugment
# ============================================================

def apply_specaugment(x, time_mask=20, freq_mask=16, num_masks=2):
    """
    x shape: [1, n_mels, time]
    """

    _, n_mels, n_time = x.shape

    for _ in range(num_masks):
        # Frequency mask
        f = random.randint(0, freq_mask)

        if f > 0 and n_mels - f > 0:
            f0 = random.randint(0, n_mels - f)
            x[:, f0:f0 + f, :] = 0

        # Time mask
        t = random.randint(0, time_mask)

        if t > 0 and n_time - t > 0:
            t0 = random.randint(0, n_time - t)
            x[:, :, t0:t0 + t] = 0

    return x


# ============================================================
# Dataset
# ============================================================

class MelFeatureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool = False):
        self.df = df.reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        feature_path = row["feature_path"]

        class_id = int(row["class_id"])
        letter_id = int(row["letter_id"])
        vowel_id = int(row["vowel_id"])
        makhraj_id = int(row["makhraj_id"])

        x = np.load(feature_path).astype(np.float32)

        if x.ndim != 2:
            raise ValueError(
                f"Expected 2D Mel feature, got shape {x.shape} for {feature_path}"
            )

        # Per-sample normalization
        mean = x.mean()
        std = x.std() + 1e-6
        x = (x - mean) / std

        # [n_mels, time] -> [1, n_mels, time]
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

        if self.train and USE_SPEC_AUGMENT:
            x = apply_specaugment(
                x,
                time_mask=TIME_MASK,
                freq_mask=FREQ_MASK,
                num_masks=NUM_MASKS,
            )

        return {
            "x": x,
            "direct84": torch.tensor(class_id, dtype=torch.long),
            "letter": torch.tensor(letter_id, dtype=torch.long),
            "vowel": torch.tensor(vowel_id, dtype=torch.long),
            "makhraj": torch.tensor(makhraj_id, dtype=torch.long),
        }


# ============================================================
# Model
# ============================================================

class ResNet18MTLDirect84LetterVowelMakhraj(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = resnet18(weights=None)

        backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.encoder = backbone

        self.direct84_head = nn.Linear(in_features, NUM_DIRECT84)
        self.letter_head = nn.Linear(in_features, NUM_LETTERS)
        self.vowel_head = nn.Linear(in_features, NUM_VOWELS)
        self.makhraj_head = nn.Linear(in_features, NUM_MAKHRAJ)

    def forward(self, x):
        features = self.encoder(x)

        direct84_logits = self.direct84_head(features)
        letter_logits = self.letter_head(features)
        vowel_logits = self.vowel_head(features)
        makhraj_logits = self.makhraj_head(features)

        return {
            "direct84": direct84_logits,
            "letter": letter_logits,
            "vowel": vowel_logits,
            "makhraj": makhraj_logits,
        }


# ============================================================
# Utility
# ============================================================

def derive_84(letter_pred, vowel_pred):
    return (letter_pred * NUM_VOWELS) + vowel_pred


def compute_batch_predictions(outputs):
    direct84_pred = torch.argmax(outputs["direct84"], dim=1)
    letter_pred = torch.argmax(outputs["letter"], dim=1)
    vowel_pred = torch.argmax(outputs["vowel"], dim=1)
    makhraj_pred = torch.argmax(outputs["makhraj"], dim=1)

    derived84_pred = derive_84(letter_pred, vowel_pred)

    return direct84_pred, letter_pred, vowel_pred, makhraj_pred, derived84_pred


# ============================================================
# Train / Evaluate
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion_direct84,
    criterion_letter,
    criterion_vowel,
    criterion_makhraj,
    optimizer,
    scaler,
):
    model.train()

    total_loss = 0.0

    all_direct84_true = []
    all_direct84_pred = []

    all_letter_true = []
    all_letter_pred = []

    all_vowel_true = []
    all_vowel_pred = []

    all_makhraj_true = []
    all_makhraj_pred = []

    all_derived84_true = []
    all_derived84_pred = []

    for batch in loader:
        x = batch["x"].to(DEVICE, non_blocking=True)

        y_direct84 = batch["direct84"].to(DEVICE, non_blocking=True)
        y_letter = batch["letter"].to(DEVICE, non_blocking=True)
        y_vowel = batch["vowel"].to(DEVICE, non_blocking=True)
        y_makhraj = batch["makhraj"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
            outputs = model(x)

            loss_direct84 = criterion_direct84(outputs["direct84"], y_direct84)
            loss_letter = criterion_letter(outputs["letter"], y_letter)
            loss_vowel = criterion_vowel(outputs["vowel"], y_vowel)
            loss_makhraj = criterion_makhraj(outputs["makhraj"], y_makhraj)

            loss = (
                DIRECT84_LOSS_WEIGHT * loss_direct84
                + LETTER_LOSS_WEIGHT * loss_letter
                + VOWEL_LOSS_WEIGHT * loss_vowel
                + MAKHRAJ_LOSS_WEIGHT * loss_makhraj
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)

        direct84_pred, letter_pred, vowel_pred, makhraj_pred, derived84_pred = (
            compute_batch_predictions(outputs)
        )

        all_direct84_true.extend(y_direct84.detach().cpu().numpy())
        all_direct84_pred.extend(direct84_pred.detach().cpu().numpy())

        all_letter_true.extend(y_letter.detach().cpu().numpy())
        all_letter_pred.extend(letter_pred.detach().cpu().numpy())

        all_vowel_true.extend(y_vowel.detach().cpu().numpy())
        all_vowel_pred.extend(vowel_pred.detach().cpu().numpy())

        all_makhraj_true.extend(y_makhraj.detach().cpu().numpy())
        all_makhraj_pred.extend(makhraj_pred.detach().cpu().numpy())

        all_derived84_true.extend(y_direct84.detach().cpu().numpy())
        all_derived84_pred.extend(derived84_pred.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    metrics = {
        "loss": avg_loss,
        "acc_direct84": accuracy_score(all_direct84_true, all_direct84_pred),
        "acc_letter": accuracy_score(all_letter_true, all_letter_pred),
        "acc_vowel": accuracy_score(all_vowel_true, all_vowel_pred),
        "acc_makhraj": accuracy_score(all_makhraj_true, all_makhraj_pred),
        "acc_derived84": accuracy_score(all_derived84_true, all_derived84_pred),
    }

    return metrics


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion_direct84,
    criterion_letter,
    criterion_vowel,
    criterion_makhraj,
):
    model.eval()

    total_loss = 0.0

    all_direct84_true = []
    all_direct84_pred = []

    all_letter_true = []
    all_letter_pred = []

    all_vowel_true = []
    all_vowel_pred = []

    all_makhraj_true = []
    all_makhraj_pred = []

    all_derived84_true = []
    all_derived84_pred = []

    for batch in loader:
        x = batch["x"].to(DEVICE, non_blocking=True)

        y_direct84 = batch["direct84"].to(DEVICE, non_blocking=True)
        y_letter = batch["letter"].to(DEVICE, non_blocking=True)
        y_vowel = batch["vowel"].to(DEVICE, non_blocking=True)
        y_makhraj = batch["makhraj"].to(DEVICE, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
            outputs = model(x)

            loss_direct84 = criterion_direct84(outputs["direct84"], y_direct84)
            loss_letter = criterion_letter(outputs["letter"], y_letter)
            loss_vowel = criterion_vowel(outputs["vowel"], y_vowel)
            loss_makhraj = criterion_makhraj(outputs["makhraj"], y_makhraj)

            loss = (
                DIRECT84_LOSS_WEIGHT * loss_direct84
                + LETTER_LOSS_WEIGHT * loss_letter
                + VOWEL_LOSS_WEIGHT * loss_vowel
                + MAKHRAJ_LOSS_WEIGHT * loss_makhraj
            )

        total_loss += loss.item() * x.size(0)

        direct84_pred, letter_pred, vowel_pred, makhraj_pred, derived84_pred = (
            compute_batch_predictions(outputs)
        )

        all_direct84_true.extend(y_direct84.detach().cpu().numpy())
        all_direct84_pred.extend(direct84_pred.detach().cpu().numpy())

        all_letter_true.extend(y_letter.detach().cpu().numpy())
        all_letter_pred.extend(letter_pred.detach().cpu().numpy())

        all_vowel_true.extend(y_vowel.detach().cpu().numpy())
        all_vowel_pred.extend(vowel_pred.detach().cpu().numpy())

        all_makhraj_true.extend(y_makhraj.detach().cpu().numpy())
        all_makhraj_pred.extend(makhraj_pred.detach().cpu().numpy())

        all_derived84_true.extend(y_direct84.detach().cpu().numpy())
        all_derived84_pred.extend(derived84_pred.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    metrics = {
        "loss": avg_loss,
        "acc_direct84": accuracy_score(all_direct84_true, all_direct84_pred),
        "acc_letter": accuracy_score(all_letter_true, all_letter_pred),
        "acc_vowel": accuracy_score(all_vowel_true, all_vowel_pred),
        "acc_makhraj": accuracy_score(all_makhraj_true, all_makhraj_pred),
        "acc_derived84": accuracy_score(all_derived84_true, all_derived84_pred),

        "direct84_true": np.array(all_direct84_true),
        "direct84_pred": np.array(all_direct84_pred),

        "letter_true": np.array(all_letter_true),
        "letter_pred": np.array(all_letter_pred),

        "vowel_true": np.array(all_vowel_true),
        "vowel_pred": np.array(all_vowel_pred),

        "makhraj_true": np.array(all_makhraj_true),
        "makhraj_pred": np.array(all_makhraj_pred),

        "derived84_true": np.array(all_derived84_true),
        "derived84_pred": np.array(all_derived84_pred),
    }

    return metrics


# ============================================================
# Reports
# ============================================================

def save_classification_outputs(
    y_true,
    y_pred,
    labels,
    target_names,
    split_name,
    task_name,
):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    txt_path = REPORT_DIR / f"{split_name}_{task_name}_classification_report.txt"
    json_path = REPORT_DIR / f"{split_name}_{task_name}_classification_report.json"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_path = REPORT_DIR / f"{split_name}_{task_name}_confusion_matrix.csv"

    if target_names is not None and len(target_names) == len(labels):
        pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(cm_path)
    else:
        pd.DataFrame(cm).to_csv(cm_path, index=False)

    return report_dict


def save_all_reports(metrics, split_name):
    direct84_report = save_classification_outputs(
        y_true=metrics["direct84_true"],
        y_pred=metrics["direct84_pred"],
        labels=list(range(NUM_DIRECT84)),
        target_names=[str(i) for i in range(NUM_DIRECT84)],
        split_name=split_name,
        task_name="direct84_84",
    )

    letter_report = save_classification_outputs(
        y_true=metrics["letter_true"],
        y_pred=metrics["letter_pred"],
        labels=list(range(NUM_LETTERS)),
        target_names=[str(i) for i in range(NUM_LETTERS)],
        split_name=split_name,
        task_name="letter_28",
    )

    vowel_report = save_classification_outputs(
        y_true=metrics["vowel_true"],
        y_pred=metrics["vowel_pred"],
        labels=list(range(NUM_VOWELS)),
        target_names=["Fatha", "Kasra", "Damma"],
        split_name=split_name,
        task_name="vowel_3",
    )

    makhraj_report = save_classification_outputs(
        y_true=metrics["makhraj_true"],
        y_pred=metrics["makhraj_pred"],
        labels=list(range(NUM_MAKHRAJ)),
        target_names=MAKHRAJ_NAMES,
        split_name=split_name,
        task_name="makhraj_5",
    )

    derived84_report = save_classification_outputs(
        y_true=metrics["derived84_true"],
        y_pred=metrics["derived84_pred"],
        labels=list(range(NUM_DIRECT84)),
        target_names=[str(i) for i in range(NUM_DIRECT84)],
        split_name=split_name,
        task_name="derived84_84",
    )

    return {
        "direct84": direct84_report,
        "letter": letter_report,
        "vowel": vowel_report,
        "makhraj": makhraj_report,
        "derived84": derived84_report,
    }


# ============================================================
# Plot Curves
# ============================================================

def plot_training_curves(history_df: pd.DataFrame):
    plt.figure()
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MTL ResNet18 Direct84+Letter+Vowel+Makhraj Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RUN_DIR / "loss_curve.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(history_df["epoch"], history_df["train_acc_direct84"], label="Train Direct-84 Acc")
    plt.plot(history_df["epoch"], history_df["val_acc_direct84"], label="Val Direct-84 Acc")
    plt.plot(history_df["epoch"], history_df["val_acc_derived84"], label="Val Derived-84 Acc")
    plt.plot(history_df["epoch"], history_df["val_acc_letter"], label="Val Letter Acc")
    plt.plot(history_df["epoch"], history_df["val_acc_vowel"], label="Val Vowel Acc")
    plt.plot(history_df["epoch"], history_df["val_acc_makhraj"], label="Val Makhraj Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MTL ResNet18 Direct84+Auxiliary Heads Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RUN_DIR / "accuracy_curve.png", dpi=300)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)

    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    df = pd.read_csv(METADATA_CSV)

    required_cols = [
        "feature_path",
        "class_id",
        "letter_id",
        "vowel_id",
        "makhraj_id",
        "split",
        "is_augmented",
        "original_id",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    if val_df["is_augmented"].sum() != 0:
        raise RuntimeError("Validation set contains augmented samples.")

    if test_df["is_augmented"].sum() != 0:
        raise RuntimeError("Test set contains augmented samples.")

    print("\nDataset sizes")
    print("-------------")
    print("Train:", len(train_df))
    print("Val:", len(val_df))
    print("Test:", len(test_df))

    print("\nClass coverage")
    print("--------------")
    print("Train 84 classes:", train_df["class_id"].nunique())
    print("Val 84 classes:", val_df["class_id"].nunique())
    print("Test 84 classes:", test_df["class_id"].nunique())

    train_dataset = MelFeatureDataset(train_df, train=True)
    val_dataset = MelFeatureDataset(val_df, train=False)
    test_dataset = MelFeatureDataset(test_df, train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = ResNet18MTLDirect84LetterVowelMakhraj().to(DEVICE)

    # This was the missing definition
    criterion_direct84 = nn.CrossEntropyLoss()
    criterion_letter = nn.CrossEntropyLoss()
    criterion_vowel = nn.CrossEntropyLoss()
    criterion_makhraj = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    best_val_direct84 = -1.0
    best_model_path = RUN_DIR / "best_model.pt"

    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion_direct84=criterion_direct84,
            criterion_letter=criterion_letter,
            criterion_vowel=criterion_vowel,
            criterion_makhraj=criterion_makhraj,
            optimizer=optimizer,
            scaler=scaler,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion_direct84=criterion_direct84,
            criterion_letter=criterion_letter,
            criterion_vowel=criterion_vowel,
            criterion_makhraj=criterion_makhraj,
        )

        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc_direct84": train_metrics["acc_direct84"],
            "train_acc_derived84": train_metrics["acc_derived84"],
            "train_acc_letter": train_metrics["acc_letter"],
            "train_acc_vowel": train_metrics["acc_vowel"],
            "train_acc_makhraj": train_metrics["acc_makhraj"],
            "val_loss": val_metrics["loss"],
            "val_acc_direct84": val_metrics["acc_direct84"],
            "val_acc_derived84": val_metrics["acc_derived84"],
            "val_acc_letter": val_metrics["acc_letter"],
            "val_acc_vowel": val_metrics["acc_vowel"],
            "val_acc_makhraj": val_metrics["acc_makhraj"],
            "lr": scheduler.get_last_lr()[0],
        })

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train loss {train_metrics['loss']:.4f} "
            f"direct84 {train_metrics['acc_direct84']:.4f} "
            f"derived84 {train_metrics['acc_derived84']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} "
            f"direct84 {val_metrics['acc_direct84']:.4f} "
            f"derived84 {val_metrics['acc_derived84']:.4f} "
            f"(L {val_metrics['acc_letter']:.4f}, "
            f"V {val_metrics['acc_vowel']:.4f}, "
            f"M {val_metrics['acc_makhraj']:.4f})"
        )

        if val_metrics["acc_direct84"] > best_val_direct84:
            best_val_direct84 = val_metrics["acc_direct84"]
            torch.save(model.state_dict(), best_model_path)

    print("\nBest validation direct-84 accuracy:", best_val_direct84)
    print("Loading best model:", best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    val_metrics = evaluate(
        model=model,
        loader=val_loader,
        criterion_direct84=criterion_direct84,
        criterion_letter=criterion_letter,
        criterion_vowel=criterion_vowel,
        criterion_makhraj=criterion_makhraj,
    )

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion_direct84=criterion_direct84,
        criterion_letter=criterion_letter,
        criterion_vowel=criterion_vowel,
        criterion_makhraj=criterion_makhraj,
    )

    val_reports = save_all_reports(val_metrics, "val")
    test_reports = save_all_reports(test_metrics, "test")

    history_df = pd.DataFrame(history)
    history_df.to_csv(RUN_DIR / "history.csv", index=False)

    plot_training_curves(history_df)

    summary = {
        "model": "mtl_resnet18_direct84_letter_vowel_makhraj",
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,

        "direct84_loss_weight": DIRECT84_LOSS_WEIGHT,
        "letter_loss_weight": LETTER_LOSS_WEIGHT,
        "vowel_loss_weight": VOWEL_LOSS_WEIGHT,
        "makhraj_loss_weight": MAKHRAJ_LOSS_WEIGHT,

        "use_specaugment": USE_SPEC_AUGMENT,
        "time_mask": TIME_MASK,
        "freq_mask": FREQ_MASK,
        "num_masks": NUM_MASKS,

        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),

        "best_val_direct84_acc": float(best_val_direct84),

        "final_val_direct84_acc": float(val_metrics["acc_direct84"]),
        "final_val_derived84_acc": float(val_metrics["acc_derived84"]),
        "final_val_letter_acc": float(val_metrics["acc_letter"]),
        "final_val_vowel_acc": float(val_metrics["acc_vowel"]),
        "final_val_makhraj_acc": float(val_metrics["acc_makhraj"]),

        "final_test_direct84_acc": float(test_metrics["acc_direct84"]),
        "final_test_derived84_acc": float(test_metrics["acc_derived84"]),
        "final_test_letter_acc": float(test_metrics["acc_letter"]),
        "final_test_vowel_acc": float(test_metrics["acc_vowel"]),
        "final_test_makhraj_acc": float(test_metrics["acc_makhraj"]),

        "final_val_direct84_macro_f1": float(val_reports["direct84"]["macro avg"]["f1-score"]),
        "final_val_derived84_macro_f1": float(val_reports["derived84"]["macro avg"]["f1-score"]),
        "final_val_letter_macro_f1": float(val_reports["letter"]["macro avg"]["f1-score"]),
        "final_val_vowel_macro_f1": float(val_reports["vowel"]["macro avg"]["f1-score"]),
        "final_val_makhraj_macro_f1": float(val_reports["makhraj"]["macro avg"]["f1-score"]),

        "final_test_direct84_macro_f1": float(test_reports["direct84"]["macro avg"]["f1-score"]),
        "final_test_derived84_macro_f1": float(test_reports["derived84"]["macro avg"]["f1-score"]),
        "final_test_letter_macro_f1": float(test_reports["letter"]["macro avg"]["f1-score"]),
        "final_test_vowel_macro_f1": float(test_reports["vowel"]["macro avg"]["f1-score"]),
        "final_test_makhraj_macro_f1": float(test_reports["makhraj"]["macro avg"]["f1-score"]),
    }

    with open(RUN_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nFinal Results")
    print("-------------")
    print(f"Val Direct-84 Acc:    {val_metrics['acc_direct84']:.4f}")
    print(f"Val Direct-84 F1:     {val_reports['direct84']['macro avg']['f1-score']:.4f}")
    print(f"Val Derived-84 Acc:   {val_metrics['acc_derived84']:.4f}")
    print(f"Val Derived-84 F1:    {val_reports['derived84']['macro avg']['f1-score']:.4f}")

    print(f"Test Direct-84 Acc:   {test_metrics['acc_direct84']:.4f}")
    print(f"Test Direct-84 F1:    {test_reports['direct84']['macro avg']['f1-score']:.4f}")
    print(f"Test Derived-84 Acc:  {test_metrics['acc_derived84']:.4f}")
    print(f"Test Derived-84 F1:   {test_reports['derived84']['macro avg']['f1-score']:.4f}")

    print(f"Test Letter Acc:      {test_metrics['acc_letter']:.4f}")
    print(f"Test Vowel Acc:       {test_metrics['acc_vowel']:.4f}")
    print(f"Test Makhraj Acc:     {test_metrics['acc_makhraj']:.4f}")

    print("\nSaved:")
    print(" - best model:", best_model_path)
    print(" - history:", RUN_DIR / "history.csv")
    print(" - loss curve:", RUN_DIR / "loss_curve.png")
    print(" - accuracy curve:", RUN_DIR / "accuracy_curve.png")
    print(" - reports:", REPORT_DIR)
    print(" - summary:", RUN_DIR / "summary.json")


if __name__ == "__main__":
    main()