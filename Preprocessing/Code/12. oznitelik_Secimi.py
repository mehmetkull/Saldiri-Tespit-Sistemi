"""
Feature Selection Uygula
- Manuel seçilen feature'ları tut
- Geri kalanları drop et
- train. csv ve test.csv güncelle
"""

import pandas as pd

# ═══════════════════════════════════════════════════
# Kümülatif %90
# ═══════════════════════════════════════════════════
SELECTED_FEATURES = [
    " Bwd Packet Length Std",
    " Avg Bwd Segment Size",
    " Packet Length Variance",
    " Bwd Packet Length Mean",
    " Packet Length Std",
    " Average Packet Size",
    " Max Packet Length",
    "Bwd Packet Length Max",
    " Packet Length Mean",
    " Destination Port",
    "Subflow Fwd Packets",
    " Total Fwd Packets",
    "Total Length of Fwd Packets",
    " Subflow Bwd Bytes",
    " min_seg_size_forward",
    " Subflow Fwd Bytes",
    " Bwd Header Length",
    " Total Length of Bwd Packets",
    " Fwd Packet Length Max",
    " Fwd Header Length",
    " Bwd Packets/s",
    " Avg Fwd Segment Size",
    " PSH Flag Count",
    " Total Backward Packets",
    " act_data_pkt_fwd",
    " Fwd Header Length.1",
    " Subflow Bwd Packets",
    " Fwd Packet Length Mean",
    " Init_Win_bytes_backward",
    "Flow Bytes/s",
    " ACK Flag Count"
]

# daha önce yüksek korelasyonlu bulunan sütunlar
HIGH_CORR_DROP = [
    " Avg Bwd Segment Size",
    " Avg Fwd Segment Size",
    " Fwd Header Length.1",
    "Subflow Fwd Packets",
    " Subflow Bwd Packets",
    " Subflow Fwd Bytes",
    " Subflow Bwd Bytes",
    " Average Packet Size"
]

# Label sütunu (tutulacak)
TARGET_COLUMN = " Label"

# ═══════════════════════════════════════════════════
# FİNAL FEATURE LİSTESİ
# ═══════════════════════════════════════════════════
# Manuel drop'ları çıkar
FINAL_FEATURES = [f for f in SELECTED_FEATURES if f not in HIGH_CORR_DROP]

# Label ekle
COLUMNS_TO_KEEP = FINAL_FEATURES + [TARGET_COLUMN]

print("=" * 70)
print("   FEATURE SELECTION UYGULA")
print("=" * 70)
print(f"\nSeçilen feature sayısı: {len(SELECTED_FEATURES)}")
print(f"Manuel drop:  {len(HIGH_CORR_DROP)}")
print(f"Final feature sayısı: {len(FINAL_FEATURES)}")
print(f"Label ile toplam sütun:  {len(COLUMNS_TO_KEEP)}\n")

# ═══════════════════════════════════════════════════
# TRAIN.CSV
# ═══════════════════════════════════════════════════
print("📂 train.csv işleniyor...")
df_train = pd.read_csv("train.csv")
original_cols_train = len(df_train.columns)

df_train_selected = df_train[COLUMNS_TO_KEEP]

df_train_selected.to_csv("train_selected.csv", index=False)
print(f"✓ {original_cols_train} → {len(df_train_selected. columns)} sütun")
print(f"✓ train_selected.csv kaydedildi ({len(df_train_selected):,} satır)\n")

# ═══════════════════════════════════════════════════
# TEST.CSV
# ═══════════════════════════════════════════════════
print("📂 test.csv işleniyor...")
df_test = pd.read_csv("test.csv")
original_cols_test = len(df_test.columns)

df_test_selected = df_test[COLUMNS_TO_KEEP]

df_test_selected.to_csv("test_selected.csv", index=False)
print(f"✓ {original_cols_test} → {len(df_test_selected.columns)} sütun")
print(f"✓ test_selected. csv kaydedildi ({len(df_test_selected):,} satır)\n")

# ═══════════════════════════════════════════════════
# ÖZET
# ═══════════════════════════════════════════════════
print("=" * 70)
print("   ÖZET")
print("=" * 70)
print(f"✓ {len(FINAL_FEATURES)} feature + Label")
print(f"✓ train_selected.csv:  {len(df_train_selected):,} satır")
print(f"✓ test_selected.csv: {len(df_test_selected):,} satır")
print("\n📋 Tutulan feature'lar:")
for i, feat in enumerate(FINAL_FEATURES, 1):
    print(f"  {i: 2d}. {feat}")
print("\n❌ Drop edilen feature'lar:")
for feat in HIGH_CORR_DROP:
    print(f"  - {feat}")
print("=" * 70)