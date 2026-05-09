import random
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from config_safe import (
    SPLIT_METADATA_CSV,
    FINAL_METADATA_CSV,
    AUG_AUDIO_DIR,
    FEATURES_DIR,
    SAMPLE_RATE,
    MAX_DURATION_SECONDS,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    AUGMENTATIONS_PER_TRAIN_SAMPLE,
    NOISE_SNR_MIN,
    NOISE_SNR_MAX,
    TIME_SHIFT_MIN_MS,
    TIME_SHIFT_MAX_MS,
    TIME_STRETCH_MIN,
    TIME_STRETCH_MAX,
    PITCH_SHIFT_MIN,
    PITCH_SHIFT_MAX,
    GAIN_MIN_DB,
    GAIN_MAX_DB,
    RANDOM_SEED,
)


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def ensure_dirs():
    AUG_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)


def load_audio(audio_path: str) -> np.ndarray:
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=25)

    # Pad or trim
    max_len = int(SAMPLE_RATE * MAX_DURATION_SECONDS)

    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)))

    # Normalize
    peak = np.max(np.abs(y)) + 1e-9
    y = y / peak

    return y.astype(np.float32)


def add_noise(y: np.ndarray) -> np.ndarray:
    snr_db = random.uniform(NOISE_SNR_MIN, NOISE_SNR_MAX)

    noise = np.random.randn(len(y)).astype(np.float32)

    signal_power = np.mean(y ** 2) + 1e-9
    noise_power = signal_power / (10 ** (snr_db / 10))

    noise = noise * np.sqrt(noise_power / (np.mean(noise ** 2) + 1e-9))

    return y + noise


def time_shift(y: np.ndarray) -> np.ndarray:
    shift_ms = random.randint(TIME_SHIFT_MIN_MS, TIME_SHIFT_MAX_MS)
    shift_samples = int(SAMPLE_RATE * shift_ms / 1000)

    direction = random.choice([-1, 1])
    shift_samples = direction * shift_samples

    return np.roll(y, shift_samples)


def time_stretch(y: np.ndarray) -> np.ndarray:
    rate = random.uniform(TIME_STRETCH_MIN, TIME_STRETCH_MAX)

    y2 = librosa.effects.time_stretch(y, rate=rate)

    max_len = int(SAMPLE_RATE * MAX_DURATION_SECONDS)

    if len(y2) > max_len:
        y2 = y2[:max_len]
    else:
        y2 = np.pad(y2, (0, max_len - len(y2)))

    return y2.astype(np.float32)


def pitch_shift(y: np.ndarray) -> np.ndarray:
    n_steps = random.uniform(PITCH_SHIFT_MIN, PITCH_SHIFT_MAX)

    return librosa.effects.pitch_shift(
        y,
        sr=SAMPLE_RATE,
        n_steps=n_steps
    ).astype(np.float32)


def gain_change(y: np.ndarray) -> np.ndarray:
    gain_db = random.uniform(GAIN_MIN_DB, GAIN_MAX_DB)
    gain = 10 ** (gain_db / 20)

    return y * gain


def augment_audio(y: np.ndarray):
    aug_type = random.choice([
        "noise",
        "shift",
        "stretch",
        "pitch",
        "gain",
        "combo",
    ])

    if aug_type == "noise":
        y2 = add_noise(y)

    elif aug_type == "shift":
        y2 = time_shift(y)

    elif aug_type == "stretch":
        y2 = time_stretch(y)

    elif aug_type == "pitch":
        y2 = pitch_shift(y)

    elif aug_type == "gain":
        y2 = gain_change(y)

    else:
        y2 = add_noise(y)
        y2 = time_shift(y2)
        y2 = gain_change(y2)

    y2 = np.clip(y2, -1.0, 1.0)

    return y2.astype(np.float32), aug_type


def audio_to_logmel(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    logmel = librosa.power_to_db(mel, ref=np.max)

    return logmel.astype(np.float32)


def save_feature(y: np.ndarray, feature_id: str) -> str:
    split_dir = FEATURES_DIR / feature_id[:5]
    split_dir.mkdir(parents=True, exist_ok=True)

    feature_path = FEATURES_DIR / f"{feature_id}.npy"

    logmel = audio_to_logmel(y)
    np.save(feature_path, logmel)

    return str(feature_path)


def process_original(row, variant_name: str, y: np.ndarray, audio_path: str):
    feature_id = f"{row['split']}_{row['original_id']}_{variant_name}"

    feature_path = FEATURES_DIR / f"{feature_id}.npy"

    logmel = audio_to_logmel(y)
    np.save(feature_path, logmel)

    out_row = row.to_dict()
    out_row["variant"] = variant_name
    out_row["is_augmented"] = 0
    out_row["processed_audio_path"] = audio_path
    out_row["feature_path"] = str(feature_path)

    return out_row


def process_augmented(row, aug_index: int, y_aug: np.ndarray, aug_name: str):
    feature_id = f"{row['split']}_{row['original_id']}_aug{aug_index}_{aug_name}"

    wav_path = AUG_AUDIO_DIR / f"{feature_id}.wav"
    feature_path = FEATURES_DIR / f"{feature_id}.npy"

    sf.write(wav_path, y_aug, SAMPLE_RATE)

    logmel = audio_to_logmel(y_aug)
    np.save(feature_path, logmel)

    out_row = row.to_dict()
    out_row["variant"] = f"aug{aug_index}_{aug_name}"
    out_row["is_augmented"] = 1
    out_row["processed_audio_path"] = str(wav_path)
    out_row["feature_path"] = str(feature_path)

    return out_row


def main():
    ensure_dirs()

    df = pd.read_csv(SPLIT_METADATA_CSV)

    required_cols = [
        "original_id",
        "audio_path",
        "split",
        "class_id",
        "letter_id",
        "vowel_id",
        "makhraj_id",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    output_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        audio_path = row["audio_path"]

        y = load_audio(audio_path)

        # Always save original feature
        output_rows.append(
            process_original(
                row=row,
                variant_name="orig",
                y=y,
                audio_path=audio_path,
            )
        )

        # Augment train only
        if row["split"] == "train":
            for aug_index in range(AUGMENTATIONS_PER_TRAIN_SAMPLE):
                y_aug, aug_name = augment_audio(y)

                output_rows.append(
                    process_augmented(
                        row=row,
                        aug_index=aug_index,
                        y_aug=y_aug,
                        aug_name=aug_name,
                    )
                )

    out_df = pd.DataFrame(output_rows)

    # Final leakage check
    # Validation/test must not contain augmented samples
    bad_eval_aug = out_df[
        (out_df["split"].isin(["val", "test"])) &
        (out_df["is_augmented"] == 1)
    ]

    if len(bad_eval_aug) > 0:
        raise RuntimeError("Validation/test contain augmented samples. This is not allowed.")

    # Each original_id must belong to only one split
    split_check = out_df.groupby("original_id")["split"].nunique()
    leaked = split_check[split_check > 1]

    if len(leaked) > 0:
        raise RuntimeError("Leakage detected: same original_id appears in multiple splits.")

    out_df.to_csv(FINAL_METADATA_CSV, index=False)

    print("\nFinal feature metadata summary")
    print("------------------------------")
    print(out_df["split"].value_counts())
    print("\nAugmentation by split:")
    print(pd.crosstab(out_df["split"], out_df["is_augmented"]))

    print("\nSaved:")
    print(FINAL_METADATA_CSV)


if __name__ == "__main__":
    main()