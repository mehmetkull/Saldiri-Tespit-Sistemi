"""
Veri Temizleme
- Özel karakterler, inf, NaN temizliği
- Veri tipi düzeltme
- Duplicate silme
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("   VERİ TEMİZLEME")
print("=" * 70)

# ═══════════════════════════════════════════════════
# VERİ YÜKLEME
# ═══════════════════════════════════════════════════
df = pd.read_csv('merged_data.csv')
original_shape = df.shape
print(f"\n✓ Yüklendi: {df.shape[0]: ,} satır, {df. shape[1]} sütun\n")

log = [f"VERİ TEMİZLEME RAPORU", 
       f"Başlangıç: {df.shape[0]:,} satır, {df.shape[1]} sütun", 
       ""]

# ═══════════════════════════════════════════════════
# 1. ÖZEL KARAKTERLER → NaN
# ═══════════════════════════════════════════════════
print("1. Özel karakterler → NaN")
df.replace(['?', 'NA', 'N/A', 'null', 'NULL', '', ' ', 'None', '-', 'nan'], np.nan, inplace=True)
log.append("✓ Özel karakterler NaN'a çevrildi")

# ═══════════════════════════════════════════════════
# 2. INF/-INF → NaN
# ═══════════════════════════════════════════════════
print("2. Inf/-inf → NaN")
inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
print(f"   {inf_count:,} inf temizlendi")
log.append(f"✓ {inf_count:,} inf/-inf NaN'a çevrildi")

# ═══════════════════════════════════════════════════
# 3. VERİ TİPİ DÜZELTME
# ═══════════════════════════════════════════════════
print("3. Veri tipi düzeltme")
converted = 0
for col in df.columns:
    if df[col]. dtype == 'object' and col != ' Label':
        df[col] = pd.to_numeric(df[col], errors='coerce')
        converted += 1

print(f"   {converted} sütun numeric'e çevrildi")
log.append(f"✓ {converted} sütun numeric'e çevrildi")

# ═══════════════════════════════════════════════════
# 4. NaN DOLDURMA
# ═══════════════════════════════════════════════════
print("4. NaN doldurma")
nan_count = df.isnull().sum().sum()

for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype in ['int64', 'float64']: 
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

print(f"   {nan_count:,} NaN dolduruldu")
log.append(f"✓ {nan_count: ,} NaN dolduruldu (median/mode)")

# ═══════════════════════════════════════════════════
# 5. DUPLICATE SİLME
# ═══════════════════════════════════════════════════
print("5. Duplicate silme")
before = len(df)
df.drop_duplicates(inplace=True)
deleted = before - len(df)
print(f"   {deleted:,} duplicate silindi")
log.append(f"✓ {deleted:,} duplicate silindi")

# ═══════════════════════════════════════════════════
# KAYIT
# ═══════════════════════════════════════════════════
log.append("")
log.append(f"SON:  {df.shape[0]:,} satır, {df.shape[1]} sütun")
log.append(f"Silinen satır: {original_shape[0] - df.shape[0]: ,}")

with open('temizleme_raporu.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(log))

df.to_csv('cleaned_data.csv', index=False)

print(f"\n✓ {original_shape} → {df.shape}")
print(f"✓ cleaned_data.csv kaydedildi")
print(f"✓ temizleme_raporu.txt kaydedildi")
print("=" * 70)