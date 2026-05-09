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

RUN_DIR = PROJECT_ROOT / "runs_torch" / "safe_single_task_resnet18_vowel"
REPORT_DIR = RUN_DIR / "reports"

NUM_CLASSES = 3
CLASS_NAMES = ["Fatha", "Kasra", "Damma"]

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
        f = random.randint(0, freq_mask)

        if f > 0 and n_mels - f > 0:
            f0 = random.randint(0, n_mels - f)
            x[:, f0:f0 + f, :] = 0

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
        vowel_id = int(row["vowel_id"])

        x = np.load(feature_path).astype(np.float32)

        if x.ndim != 2:
            raise ValueError(
                f"Expected 2D Mel feature, got shape {x.shape} for {feature_path}"
            )

        mean = x.mean()
        std = x.std() + 1e-6
        x = (x - mean) / std

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

        if self.train and USE_SPEC_AUGMENT:
            x = apply_specaugment(
                x,
                time_mask=TIME_MASK,
                freq_mask=FREQ_MASK,
                num_masks=NUM_MASKS,
            )

        y = torch.tensor(vowel_id, dtype=torch.long)

        return x, y


# ============================================================
# Model
# ============================================================

class ResNet18SingleTaskVowel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.backbone = resnet18(weights=None)

        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# ============================================================
# Train / Evaluate
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)

    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)

    return avg_loss, acc, np.array(all_targets), np.array(all_preds)


# ============================================================
# Plot curves
# ============================================================

def plot_training_curves(history_df: pd.DataFrame):
    plt.figure()
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Single-Task ResNet18 Vowel Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RUN_DIR / "loss_curve.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(history_df["epoch"], history_df["train_acc"], label="Train Accuracy")
    plt.plot(history_df["epoch"], history_df["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Single-Task ResNet18 Vowel Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RUN_DIR / "accuracy_curve.png", dpi=300)
    plt.close()


# ============================================================
# Reports
# ============================================================

def save_reports(y_true, y_pred, split_name: str):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    labels = list(range(NUM_CLASSES))

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=CLASS_NAMES,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    report_txt_path = REPORT_DIR / f"{split_name}_vowel_3_classification_report.txt"
    report_json_path = REPORT_DIR / f"{split_name}_vowel_3_classification_report.json"

    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    cm_path = REPORT_DIR / f"{split_name}_vowel_3_confusion_matrix.csv"
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(cm_path)

    return report_dict


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
        "vowel_id",
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

    print("\nVowel distribution by split")
    print("---------------------------")
    print(pd.crosstab(df["vowel_id"], df["split"]))

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

    model = ResNet18SingleTaskVowel(num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

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

    best_val_acc = -1.0
    best_model_path = RUN_DIR / "best_model.pt"

    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
        )

        val_loss, val_acc, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
        )

        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        })

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    print("\nBest validation accuracy:", best_val_acc)
    print("Loading best model:", best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    val_loss, val_acc, val_true, val_pred = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
    )

    test_loss, test_acc, test_true, test_pred = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
    )

    val_report = save_reports(val_true, val_pred, "val")
    test_report = save_reports(test_true, test_pred, "test")

    history_df = pd.DataFrame(history)
    history_df.to_csv(RUN_DIR / "history.csv", index=False)

    plot_training_curves(history_df)

    summary = {
        "model": "single_task_resnet18_vowel",
        "seed": SEED,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "use_specaugment": USE_SPEC_AUGMENT,
        "time_mask": TIME_MASK,
        "freq_mask": FREQ_MASK,
        "num_masks": NUM_MASKS,
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "best_val_acc": float(best_val_acc),
        "final_val_acc": float(val_acc),
        "final_test_acc": float(test_acc),
        "final_val_macro_f1": float(val_report["macro avg"]["f1-score"]),
        "final_test_macro_f1": float(test_report["macro avg"]["f1-score"]),
        "final_val_weighted_f1": float(val_report["weighted avg"]["f1-score"]),
        "final_test_weighted_f1": float(test_report["weighted avg"]["f1-score"]),
    }

    with open(RUN_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nFinal Results")
    print("-------------")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Macro-F1: {val_report['macro avg']['f1-score']:.4f}")
    print(f"Test Accuracy:       {test_acc:.4f}")
    print(f"Test Macro-F1:       {test_report['macro avg']['f1-score']:.4f}")

    print("\nSaved:")
    print(" - best model:", best_model_path)
    print(" - history:", RUN_DIR / "history.csv")
    print(" - loss curve:", RUN_DIR / "loss_curve.png")
    print(" - accuracy curve:", RUN_DIR / "accuracy_curve.png")
    print(" - reports:", REPORT_DIR)
    print(" - summary:", RUN_DIR / "summary.json")


if __name__ == "__main__":
    main()