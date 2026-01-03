"""
Grid Search + Model Kaydetme
- Hassas parametrelerde ince ayar
- En iyi modeli kaydet
- Reproducibility için tüm bilgileri kaydet
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report

# ═══════════════════════════════════════════════════
# AYARLAR
# ═══════════════════════════════════════════════════
TRAIN_FILE = "/home/azureuser/IDS_Project/test/train_selected.csv"
TEST_FILE = "/home/azureuser/IDS_Project/test/test_selected.csv"
RESULTS_DIR = Path("results")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")  # Model klasörü
TARGET_COLUMN = " Label"
RANDOM_STATE = 42
N_FOLDS = 5

RESULTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)  # Model klasörü oluştur

# ═══════════════════════════════════════════════════
# GRID PARAMETRELERİ
# ═══════════════════════════════════════════════════
param_grid = {
    'n_estimators': [150],
    'max_depth': [19, 20, 21],
    'min_samples_split':  [2, 3],
    'min_samples_leaf': [1],
    'max_features': ['log2'],
    'class_weight': [None]
}

total_combinations = np.prod([len(v) for v in param_grid.values()])

print("=" * 70)
print("   GRID SEARCH + MODEL KAYDETME")
print("=" * 70)
print(f"\nTarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nStrateji:  Sadece hassas parametrelerde ince ayar")
print(f"\nSabit parametreler (RandomSearch'ten):")
print(f"  - n_estimators:      150")
print(f"  - max_features:     log2")
print(f"  - class_weight:     None")
print(f"  - min_samples_leaf: 1")

print(f"\nFine-tune edilecek:")
print(f"  - max_depth:        {param_grid['max_depth']}")
print(f"  - min_samples_split: {param_grid['min_samples_split']}")

print(f"\nToplam kombinasyon:    {total_combinations}")
print(f"Tahmini süre:        ~{total_combinations * N_FOLDS * 3:.0f} dakika ({total_combinations * N_FOLDS * 3 / 60:.1f} saat)")
print()

# ═══════════════════════════════════════════════════
# VERİ YÜKLEME
# ═══════════════════════════════════════════════════
print("📂 Veri yükleniyor...")
df_train = pd.read_csv(TRAIN_FILE)
df_test = pd.read_csv(TEST_FILE)

X_train = df_train.drop(columns=[TARGET_COLUMN])
y_train = df_train[TARGET_COLUMN].squeeze()
X_test = df_test.drop(columns=[TARGET_COLUMN])
y_test = df_test[TARGET_COLUMN].squeeze()

print(f"✓ Train:   {len(X_train):,} satır, {len(X_train.columns)} feature")
print(f"✓ Test:   {len(X_test):,} satır")
print()

# ═══════════════════════════════════════════════════
# BASELINE (RandomSearch En İyi)
# ═══════════════════════════════════════════════════
print("=" * 70)
print("   BASELINE (RandomSearch En İyi)")
print("=" * 70)

baseline_params = {
    'n_estimators': 150,
    'max_depth': 20,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features':  'log2',
    'class_weight': None,
    'random_state':  RANDOM_STATE,
    'n_jobs': -1
}

print("\n⏳ Baseline eğitiliyor...")
baseline_start = time.time()

baseline = RandomForestClassifier(**baseline_params)
baseline.fit(X_train, y_train)
y_pred_baseline = baseline.predict(X_test)

baseline_time = time.time() - baseline_start

baseline_f1 = f1_score(y_test, y_pred_baseline, average='macro')
baseline_recall = recall_score(y_test, y_pred_baseline, average='macro')
baseline_precision = precision_score(y_test, y_pred_baseline, average='macro')
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)

print(f"✓ Tamamlandı ({baseline_time:.1f}s)")
print(f"\nRandomSearch En İyi Test Sonuçları:")
print(f"  F1-macro:    {baseline_f1:.6f}")
print(f"  Recall:     {baseline_recall:.6f}")
print(f"  Precision:  {baseline_precision:.6f}")
print(f"  Accuracy:   {baseline_accuracy:.6f}")

# Baseline modeli kaydet
baseline_model_file = MODELS_DIR / "baseline_random_search_model.pkl"
joblib.dump(baseline, baseline_model_file)
print(f"\n✓ Baseline model kaydedildi: {baseline_model_file}")
print()

# ═══════════════════════════════════════════════════
# GRID SEARCH
# ═══════════════════════════════════════════════════
print("=" * 70)
print("   GRID SEARCH")
print("=" * 70)

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=1,
        max_features='log2',
        class_weight=None,
        random_state=RANDOM_STATE,
        n_jobs=1
    ),
    param_grid={
        'max_depth': param_grid['max_depth'],
        'min_samples_split':  param_grid['min_samples_split']
    },
    cv=N_FOLDS,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

print("\n⏳ GridSearchCV başladı...")
print("-" * 70)

search_start = time.time()
grid_search.fit(X_train, y_train)
search_time = time.time() - search_start

print("-" * 70)
print(f"✓ Tamamlandı ({search_time/60:.1f} dakika)")
print()

# ═══════════════════════════════════════════════════
# EN İYİ MODEL
# ═══════════════════════════════════════════════════
print("=" * 70)
print("   EN İYİ MODEL")
print("=" * 70)

best_params = grid_search.best_params_
best_cv_score = grid_search.best_score_
best_model = grid_search.best_estimator_

print(f"\n📊 GridSearch En İyi CV F1: {best_cv_score:.6f}")
print(f"📊 RandomSearch'ten iyileşme: {(best_cv_score - 0.993738)*100:+.4f}%")

print(f"\n✅ En İyi Parametreler:")
print("-" * 70)
print(f"  max_depth:           {best_params['max_depth']}")
print(f"  min_samples_split:  {best_params['min_samples_split']}")
print(f"  n_estimators:       150 (sabit)")
print(f"  min_samples_leaf:   1 (sabit)")
print(f"  max_features:       log2 (sabit)")
print(f"  class_weight:       None (sabit)")
print()

# ═══════════════════════════════════════════════════
# TEST DEĞERLENDİRME
# ═══════════════════════════════════════════════════
print("=" * 70)
print("   TEST SETİ DEĞERLENDİRME")
print("=" * 70)

y_pred_tuned = best_model.predict(X_test)

tuned_f1 = f1_score(y_test, y_pred_tuned, average='macro')
tuned_recall = recall_score(y_test, y_pred_tuned, average='macro')
tuned_precision = precision_score(y_test, y_pred_tuned, average='macro')
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)

print(f"\n📊 Karşılaştırma:")
print("-" * 70)
print(f"{'Metrik':<15} {'RandomSearch':<15} {'GridSearch':<15} {'İyileşme':<12}")
print("-" * 70)
print(f"{'F1-macro':<15} {baseline_f1:<15.6f} {tuned_f1:<15.6f} {(tuned_f1-baseline_f1)*100:+.4f}%")
print(f"{'Recall':<15} {baseline_recall:<15.6f} {tuned_recall:<15.6f} {(tuned_recall-baseline_recall)*100:+.4f}%")
print(f"{'Precision':<15} {baseline_precision:<15.6f} {tuned_precision:<15.6f} {(tuned_precision-baseline_precision)*100:+.4f}%")
print(f"{'Accuracy':<15} {baseline_accuracy:<15.6f} {tuned_accuracy:<15.6f} {(tuned_accuracy-baseline_accuracy)*100:+.4f}%")

improvement = (tuned_f1 - baseline_f1) * 100
print()
if improvement > 0.01:
    print(f"✅ Anlamlı iyileşme: +{improvement:.4f}%")
    decision = "GridSearch kazandı!  Bu modeli kullan."
elif improvement > 0:
    print(f"⚠️  Minimal iyileşme: +{improvement:.4f}%")
    decision = "GridSearch minimal fayda sağladı."
else:
    print(f"❌ İyileşme yok: {improvement:+.4f}%")
    decision = "RandomSearch yeterliydi."

print(f"\n📋 Classification Report (GridSearch):")
print("-" * 70)
print(classification_report(y_test, y_pred_tuned, digits=4))

# ═══════════════════════════════════════════════════
# MODEL KAYDETME
# ═══════════════════════════════════════════════════
print("=" * 70)
print("   MODEL KAYDETME")
print("=" * 70)

# ✅ 1. GridSearch en iyi modeli
grid_model_file = MODELS_DIR / "grid_search_best_model.pkl"
joblib.dump(best_model, grid_model_file)
print(f"\n✓ GridSearch en iyi model:  {grid_model_file}")

# ✅ 2. Tüm GridSearchCV objesi (tüm kombinasyonlar)
grid_cv_file = MODELS_DIR / "grid_search_cv_object.pkl"
joblib.dump(grid_search, grid_cv_file)
print(f"✓ GridSearchCV objesi:       {grid_cv_file}")

# ✅ 3. Final model seçimi (en iyi olan)
if improvement > 0:
    final_model = best_model
    final_model_source = "GridSearch"
    final_f1 = tuned_f1
else:
    final_model = baseline
    final_model_source = "RandomSearch (GridSearch iyileştirme sağlamadı)"
    final_f1 = baseline_f1

final_model_file = MODELS_DIR / "final_model.pkl"
joblib.dump(final_model, final_model_file)
print(f"✓ Final model:               {final_model_file}")
print(f"  Kaynak:  {final_model_source}")
print(f"  Test F1: {final_f1:.6f}")

# ✅ 4. Model metadata
model_metadata = {
    "model_file": str(final_model_file),
    "source": final_model_source,
    "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "test_f1_macro": float(final_f1),
    "parameters": final_model.get_params(),
    "feature_count": len(X_train.columns),
    "train_samples": len(X_train),
    "test_samples": len(X_test)
}

metadata_file = MODELS_DIR / "final_model_metadata.json"
with open(metadata_file, 'w') as f:
    json.dump(model_metadata, f, indent=2)
print(f"✓ Model metadata:           {metadata_file}")

print()

# ═══════════════════════════════════════════════════
# TÜM KOMBİNASYONLAR
# ═══════════════════════════════════════════════════
print("=" * 70)
print("   TÜM KOMBİNASYONLAR")
print("=" * 70)

cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results_sorted = cv_results.sort_values('rank_test_score')

print(f"\n{'Rank':<6} {'max_depth':<12} {'min_split':<12} {'CV F1':<14} {'Std':<10}")
print("-" * 70)
for _, row in cv_results_sorted.iterrows():
    print(f"{int(row['rank_test_score']):<6} "
          f"{str(row['param_max_depth']):<12} "
          f"{str(row['param_min_samples_split']):<12} "
          f"{row['mean_test_score']:.6f}      "
          f"±{row['std_test_score']:.6f}")

print()

# ═══════════════════════════════════════════════════
# SONUÇLARI KAYDET
# ═══════════════════════════════════════════════════
print("=" * 70)
print("   SONUÇLARI KAYDET")
print("=" * 70)

results = {
    "experiment":  "Grid Search - Ultra Minimal",
    "date":  datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "strategy": "Fine-tune only max_depth and min_samples_split",
    
    "grid_config": {
        "total_combinations": int(total_combinations),
        "total_fits": int(total_combinations * N_FOLDS),
        "search_time_minutes": round(search_time / 60, 2),
        "avg_time_per_fit_minutes": round(search_time / (total_combinations * N_FOLDS), 2)
    },
    
    "baseline_random_search": {
        "cv_f1": 0.993738,
        "test_f1_macro": float(baseline_f1),
        "test_recall": float(baseline_recall),
        "test_precision": float(baseline_precision),
        "test_accuracy": float(baseline_accuracy),
        "params": baseline_params,
        "model_file": str(baseline_model_file)
    },
    
    "grid_search_best":  {
        "cv_f1": float(best_cv_score),
        "test_f1_macro": float(tuned_f1),
        "test_recall": float(tuned_recall),
        "test_precision": float(tuned_precision),
        "test_accuracy": float(tuned_accuracy),
        "params": {
            "max_depth": int(best_params['max_depth']),
            "min_samples_split": int(best_params['min_samples_split']),
            "n_estimators":  150,
            "min_samples_leaf": 1,
            "max_features": "log2",
            "class_weight": None
        },
        "improvement_cv_%": float((best_cv_score - 0.993738) * 100),
        "improvement_test_%": float((tuned_f1 - baseline_f1) * 100),
        "model_file": str(grid_model_file)
    },
    
    "final_model": {
        "source": final_model_source,
        "test_f1_macro": float(final_f1),
        "model_file": str(final_model_file),
        "metadata_file": str(metadata_file),
        "decision": decision
    },
    
    "all_combinations": [
        {
            "rank": int(row['rank_test_score']),
            "max_depth": int(row['param_max_depth']),
            "min_samples_split": int(row['param_min_samples_split']),
            "cv_f1_mean": float(row['mean_test_score']),
            "cv_f1_std": float(row['std_test_score'])
        }
        for _, row in cv_results_sorted.iterrows()
    ],
    
    "files":  {
        "baseline_model": str(baseline_model_file),
        "grid_best_model": str(grid_model_file),
        "grid_cv_object": str(grid_cv_file),
        "final_model": str(final_model_file),
        "final_metadata": str(metadata_file)
    }
}

json_file = RESULTS_DIR / "grid_search_minimal_results.json"
with open(json_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ {json_file}")

# CSV
csv_file = RESULTS_DIR / "grid_search_cv_details.csv"
cv_results.to_csv(csv_file, index=False)
print(f"✓ {csv_file}")

print()

# ═══════════════════════════════════════════════════
# ÖZET
# ═══════════════════════════════════════════════════
print("=" * 70)
print("   ÖZET")
print("=" * 70)
print(f"✓ {total_combinations} kombinasyon × {N_FOLDS}-fold = {total_combinations * N_FOLDS} fit")
print(f"✓ Gerçek süre: {search_time/60:.1f} dakika")
print(f"✓ Ortalama fit süresi: {search_time/(total_combinations * N_FOLDS):.2f} dakika")
print(f"✓ Test F1 iyileşme: {(tuned_f1-baseline_f1)*100:+.4f}%")

print(f"\n📁 Kaydedilen dosyalar:")
print("-" * 70)
print(f"Modeller:")
print(f"  ├─ {baseline_model_file.name}")
print(f"  ├─ {grid_model_file.name}")
print(f"  ├─ {grid_cv_file.name}")
print(f"  └─ {final_model_file.name} ⭐ (kullanılacak)")

print(f"\nRaporlar:")
print(f"  ├─ {json_file.name}")
print(f"  ├─ {csv_file.name}")
print(f"  └─ {metadata_file.name}")

print(f"\n🎯 Karar:  {decision}")
print(f"🎯 Final Model: {final_model_file}")
print(f"🎯 Test F1-macro: {final_f1:.6f}")
print("=" * 70)