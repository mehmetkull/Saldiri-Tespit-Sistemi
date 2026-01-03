"""
Train/Test Split Scripti: Veriyi yükler, 80/20 split yapar, train ve test setlerini ayrı csv olarak kaydeder, rapor çıkarır.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# ═══════════════════════════════════════════════════
# AYARLAR
# ═══════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent.parent  # proje kökü (training)
DATA_FILE = BASE_DIR / "shuffled.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_COLUMN = " Label"
SPLITS_DIR = BASE_DIR / "splits"

# ═══════════════════════════════════════════════════
# VERİ YÜKLEME
# ═══════════════════════════════════════════════════
print("=" * 70)
print("   TRAIN/TEST SPLIT")
print("=" * 70)
print(f"\n📂 Veri yükleniyor:  {DATA_FILE}")

df = pd.read_csv(DATA_FILE)
print(f"✓ Yüklendi:  {len(df):,} satır, {len(df. columns)} kolon\n")

# ═══════════════════════════════════════════════════
# ETİKET DAĞILIMI
# ═══════════════════════════════════════════════════
print("🎯 ETIKET DAĞILIMI (Tüm Veri)")
print("-" * 70)
label_counts = df[TARGET_COLUMN].value_counts()
label_pct = (label_counts / len(df) * 100)

for label, count in label_counts.items():
    print(f"{str(label):20s}: {count:>8,} ({label_pct[label]: >6.2f}%)")
print(f"{'-' * 70}")
print(f"{'Toplam':20s}:  {len(df):>8,} (100.00%)\n")

# Class imbalance kontrolü
imbalance_ratio = label_counts. max() / label_counts.min()
if imbalance_ratio > 10:
    print(f"⚠️  Class Imbalance:  {imbalance_ratio:.1f}x → SMOTE önerilir\n")

# ═══════════════════════════════════════════════════
# TRAIN/TEST SPLIT
# ═══════════════════════════════════════════════════
print("✂️  SPLIT YAPILIYOR")
print("-" * 70)
print(f"Oran: {int((1-TEST_SIZE)*100)}% / {int(TEST_SIZE*100)}%")
print(f"Strateji:  Stratified")
print(f"Random State: {RANDOM_STATE}\n")

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Train ve test setlerini proje köküne kaydet
train_csv_path = BASE_DIR / "train.csv"
test_csv_path = BASE_DIR / "test.csv"
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)
train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

# ═══════════════════════════════════════════════════
# TRAIN SET
# ═══════════════════════════════════════════════════
print("📦 TRAIN SET")
print("-" * 70)
print(f"Satır:  {len(X_train):,} ({len(X_train)/len(df)*100:.2f}%)\n")

train_label_counts = y_train. value_counts()
train_label_pct = (train_label_counts / len(y_train) * 100)

for label, count in train_label_counts.items():
    print(f"  {str(label):18s}: {count:>8,} ({train_label_pct[label]:>6.2f}%)")

# ═══════════════════════════════════════════════════
# TEST SET
# ═══════════════════════════════════════════════════
print(f"\n📦 TEST SET")
print("-" * 70)
print(f"Satır: {len(X_test):,} ({len(X_test)/len(df)*100:.2f}%)\n")

test_label_counts = y_test.value_counts()
test_label_pct = (test_label_counts / len(y_test) * 100)

for label, count in test_label_counts. items():
    print(f"  {str(label):18s}: {count:>8,} ({test_label_pct[label]:>6.2f}%)")

# ═══════════════════════════════════════════════════
# DOĞRULAMA
# ═══════════════════════════════════════════════════
print(f"\n✅ DOĞRULAMA")
print("-" * 70)
assert len(X_train) + len(X_test) == len(df)
print("✓ Train + Test = Toplam")

assert len(set(X_train. index) & set(X_test.index)) == 0
print("✓ Overlap yok")

max_diff = max(abs(label_pct[label] - train_label_pct[label]) for label in label_pct.index)
print(f"✓ Stratification OK (max fark: {max_diff:.3f}%)")

# ═══════════════════════════════════════════════════
# METADATA KAYDET
# ═══════════════════════════════════════════════════
metadata = {
    "data_file": str(DATA_FILE),
    "random_state": RANDOM_STATE,
    "test_size":  TEST_SIZE,
    "total_rows": len(df),
    "n_features": len(X.columns),
    "target_column": TARGET_COLUMN,
    "train_csv": str(train_csv_path),
    "test_csv": str(test_csv_path),
    "train": {
        "n_samples": len(X_train),
        "label_distribution": train_label_counts.to_dict()
    },
    "test": {
        "n_samples": len(X_test),
        "label_distribution": test_label_counts.to_dict()
    },
    "class_imbalance_ratio": float(imbalance_ratio)
}

with open(SPLITS_DIR / "split_meta.json", 'w') as f:
    json.dump(metadata, f, indent=2)

# ═══════════════════════════════════════════════════
# ÖZET
# ═══════════════════════════════════════════════════
print(f"\n💾 KAYITLI DOSYALAR")
print("-" * 70)
print(f"✓ train.csv (train set, {len(X_train):,} satır)")
print(f"✓ test.csv (test set, {len(X_test):,} satır)")
print(f"✓ splits/split_meta.json (metadata)")

print("\n" + "=" * 70)
print(f"✓ Train: {len(X_train):,} ({len(X_train)/len(df)*100:.0f}%) | Test: {len(X_test):,} ({len(X_test)/len(df)*100:.0f}%)")
print(f"🎯 Sonraki adım: Baseline CV")
print("=" * 70)