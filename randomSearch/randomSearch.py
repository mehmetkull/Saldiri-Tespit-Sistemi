"""
Hyperparameter Tuning - Random Forest
- RandomizedSearchCV (5-fold CV)
- CICIDS2017 için optimize edilmiş parametreler
- Detaylı raporlama
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ═══════════════════════════════════════════════════
# AYARLAR
# ═══════════════════════════════════════════════════
TRAIN_FILE = "train_selected.csv"
TEST_FILE = "test_selected.csv"
RESULTS_DIR = Path("results")
REPORTS_DIR = Path("reports")
TARGET_COLUMN = " Label"
RANDOM_STATE = 42

# Tuning ayarları
N_ITER = 30       # Deneyecek kombinasyon sayısı
N_FOLDS = 5      # 5-fold CV 

RESULTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════
# PARAMETRE GRİDİ 
# ═══════════════════════════════════════════════════
param_grid = {
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [15, 20, 25, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':  ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

# ═══════════════════════════════════════════════════
# VERİ YÜKLEME
# ═══════════════════════════════════════════════════
print("=" * 70)
print("   HYPERPARAMETER TUNING - RANDOM FOREST")
print("=" * 70)

print(f"\n📂 Veri yükleniyor...")
df_train = pd.read_csv(TRAIN_FILE)
df_test = pd.read_csv(TEST_FILE)

X_train = df_train.drop(columns=[TARGET_COLUMN])
y_train = df_train[TARGET_COLUMN].squeeze()
X_test = df_test.drop(columns=[TARGET_COLUMN])
y_test = df_test[TARGET_COLUMN].squeeze()

print(f"✓ Train:  {len(X_train):,} satır, {len(X_train.columns)} feature")
print(f"✓ Test:    {len(X_test):,} satır, {len(X_test.columns)} feature")

# Label dağılımı
print(f"\n🎯 Label Dağılımı (Train):")
print("-" * 70)
for label, count in y_train.value_counts().sort_index().items():
    print(f"  {str(label):20s}: {count:>8,} ({count/len(y_train)*100:>6.2f}%)")
print()

# ═══════════════════════════════════════════════════
# BASELINE MODEL
# ═══════════════════════════════════════════════════
print("=" * 70)
print("   BASELINE MODEL (Default Params)")
print("=" * 70)

print("\n⏳ Baseline eğitiliyor...")
baseline_start = time.time()

baseline = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
baseline.fit(X_train, y_train)
y_pred_baseline = baseline.predict(X_test)

baseline_time = time.time() - baseline_start

baseline_f1 = f1_score(y_test, y_pred_baseline, average='macro')
baseline_recall = recall_score(y_test, y_pred_baseline, average='macro')
baseline_precision = precision_score(y_test, y_pred_baseline, average='macro')
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)

print(f"✓ Tamamlandı ({baseline_time:.1f}s)")
print(f"\n📊 Baseline Test Sonuçları:")
print(f"  F1-macro:     {baseline_f1:.4f}")
print(f"  Recall:      {baseline_recall:.4f}")
print(f"  Precision:   {baseline_precision:.4f}")
print(f"  Accuracy:    {baseline_accuracy:.4f}")
print()

# ═══════════════════════════════════════════════════
# RANDOMIZED SEARCH CV
# ═══════════════════════════════════════════════════
print("=" * 70)
print(f"   RANDOMIZED SEARCH CV ({N_FOLDS}-Fold)")
print("=" * 70)

print(f"\nParametre Grid:")
for param, values in param_grid.items():
    print(f"  {param:20s}: {values}")

total_combinations = np.prod([len(v) for v in param_grid.values()])
print(f"\nToplam olası kombinasyon:  {total_combinations:,}")
print(f"Deneyecek kombinasyon:     {N_ITER}")
print(f"CV folds:                 {N_FOLDS}")
print(f"Toplam fit sayısı:        {N_ITER * N_FOLDS}")
print()

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=False, random_state=None)

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
    param_distributions=param_grid,
    n_iter=N_ITER,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1,
    return_train_score=True
)

print("⏳ RandomizedSearchCV başladı...")
print("-" * 70)

search_start = time.time()
random_search.fit(X_train, y_train)
search_time = time.time() - search_start

print("-" * 70)
print(f"✓ Tamamlandı ({search_time/60:.1f} dakika)")
print()

# ═══════════════════════════════════════════════════
# EN İYİ PARAMETRELER
# ═══════════════════════════════════════════════════
print("=" * 70)
print("   EN İYİ PARAMETRELER")
print("=" * 70)

best_params = random_search.best_params_
best_cv_score = random_search.best_score_

print(f"\n📊 En İyi CV F1-macro: {best_cv_score:.4f}")
print(f"📊 Baseline'dan iyileşme:  {(best_cv_score - baseline_f1)*100:+.2f}%")
print(f"\n✅ En İyi Parametreler:")
print("-" * 70)
for param, value in sorted(best_params.items()):
    print(f"  {param:20s}: {value}")
print()

# ═══════════════════════════════════════════════════
# TEST SETİ DEĞERLENDİRME
# ═══════════════════════════════════════════════════
print("=" * 70)
print("   TEST SETİ DEĞERLENDİRME")
print("=" * 70)

best_model = random_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)

tuned_f1 = f1_score(y_test, y_pred_tuned, average='macro')
tuned_recall = recall_score(y_test, y_pred_tuned, average='macro')
tuned_precision = precision_score(y_test, y_pred_tuned, average='macro')
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)

print(f"\n📊 Test Set Karşılaştırma:")
print("-" * 70)
print(f"{'Metric':<15} {'Baseline':<12} {'Tuned':<12} {'İyileşme':<12}")
print("-" * 70)
print(f"{'F1-macro':<15} {baseline_f1:<12.4f} {tuned_f1:<12.4f} {(tuned_f1-baseline_f1)*100:+.2f}%")
print(f"{'Recall':<15} {baseline_recall:<12.4f} {tuned_recall:<12.4f} {(tuned_recall-baseline_recall)*100:+.2f}%")
print(f"{'Precision':<15} {baseline_precision:<12.4f} {tuned_precision:<12.4f} {(tuned_precision-baseline_precision)*100:+.2f}%")
print(f"{'Accuracy':<15} {baseline_accuracy:<12.4f} {tuned_accuracy:<12.4f} {(tuned_accuracy-baseline_accuracy)*100:+.2f}%")
print()

print("📋 Detaylı Classification Report (Tuned Model):")
print("-" * 70)
print(classification_report(y_test, y_pred_tuned, digits=4))

# ═══════════════════════════════════════════════════
# CV RESULTS ANALİZİ
# ═══════════════════════════════════════════════════
print("=" * 70)
print("   CV RESULTS ANALİZİ")
print("=" * 70)

cv_results = pd.DataFrame(random_search.cv_results_)

print(f"\nTop 5 Kombinasyon:")
print("-" * 70)
top5 = cv_results.nsmallest(5, 'rank_test_score')
for idx, row in top5.iterrows():
    print(f"\nRank #{int(row['rank_test_score'])}:")
    print(f"  CV F1-macro: {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")
    print(f"  Parametreler: {row['params']}")

print()

# ═══════════════════════════════════════════════════
# GÖRSELLEŞTİRME
# ═══════════════════════════════════════════════════
print("=" * 70)
print("   GÖRSELLEŞTİRME")
print("=" * 70)

sns.set_style("whitegrid")

# 1. Top 10 Kombinasyon Bar Chart
fig, ax = plt.subplots(figsize=(12, 7))
top10 = cv_results.nsmallest(10, 'rank_test_score')

y_pos = range(len(top10))
ax.barh(y_pos, top10['mean_test_score'], xerr=top10['std_test_score'],
        color='steelblue', capsize=4, alpha=0.7, edgecolor='black')
ax.axvline(baseline_f1, color='red', linestyle='--', linewidth=2.5,
           label=f'Baseline: {baseline_f1:.4f}', zorder=10)

ax.set_yticks(y_pos)
ax.set_yticklabels([f"Rank #{int(r)}" for r in top10['rank_test_score']], fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('F1-macro (5-fold CV)', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Hyperparameter Combinations', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(REPORTS_DIR / "random_search_top10.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ random_search_top10.png")

# 2. Parametre Etkisi - n_estimators
fig, ax = plt.subplots(figsize=(10, 6))

n_est_effect = cv_results.groupby('param_n_estimators')['mean_test_score'].agg(['mean', 'std', 'count'])
n_est_effect = n_est_effect.sort_index()

x_vals = [int(x) for x in n_est_effect.index]
ax.errorbar(x_vals, n_est_effect['mean'], yerr=n_est_effect['std'],
            marker='o', markersize=10, linewidth=2.5, capsize=5,
            color='steelblue', label='CV F1-macro')
ax.axhline(baseline_f1, color='red', linestyle='--', linewidth=2,
           label=f'Baseline: {baseline_f1:.4f}')

for x, mean, count in zip(x_vals, n_est_effect['mean'], n_est_effect['count']):
    ax.text(x, mean + 0.0002, f'n={int(count)}', ha='center', fontsize=9)

ax.set_xlabel('n_estimators', fontsize=12, fontweight='bold')
ax.set_ylabel('F1-macro (5-fold CV)', fontsize=12, fontweight='bold')
ax.set_title('Effect of n_estimators on Performance', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(REPORTS_DIR / "random_search_n_estimators.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ random_search_n_estimators.png")

# 3. Parametre Etkisi - max_depth
fig, ax = plt.subplots(figsize=(10, 6))

depth_effect = cv_results.groupby('param_max_depth')['mean_test_score'].agg(['mean', 'std', 'count'])
depth_labels = [str(d) if str(d) != 'None' else 'None' for d in depth_effect.index]

x_pos = range(len(depth_effect))
ax.bar(x_pos, depth_effect['mean'], yerr=depth_effect['std'],
       capsize=5, color='coral', alpha=0.7, edgecolor='black')
ax.axhline(baseline_f1, color='red', linestyle='--', linewidth=2,
           label=f'Baseline:  {baseline_f1:.4f}')

for i, (mean, count) in enumerate(zip(depth_effect['mean'], depth_effect['count'])):
    ax.text(i, mean + 0.0002, f'n={int(count)}', ha='center', fontsize=9)

ax.set_xticks(x_pos)
ax.set_xticklabels(depth_labels, fontsize=11)
ax.set_xlabel('max_depth', fontsize=12, fontweight='bold')
ax.set_ylabel('F1-macro (5-fold CV)', fontsize=12, fontweight='bold')
ax.set_title('Effect of max_depth on Performance', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(REPORTS_DIR / "random_search_max_depth.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ random_search_max_depth.png\n")

# ═══════════════════════════════════════════════════
# SONUÇLARI KAYDET
# ═══════════════════════════════════════════════════
print("=" * 70)
print("   SONUÇLARI KAYDET")
print("=" * 70)

# JSON
results = {
    "experiment":  "Hyperparameter Tuning - RandomizedSearchCV",
    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    
    "configuration": {
        "method": "RandomizedSearchCV",
        "n_iter": N_ITER,
        "n_folds": N_FOLDS,
        "random_state":  RANDOM_STATE,
        "scoring": "f1_macro"
    },
    
    "data": {
        "train_file":  TRAIN_FILE,
        "test_file": TEST_FILE,
        "n_train":  len(X_train),
        "n_test": len(X_test),
        "n_features": len(X_train.columns),
        "label_distribution": y_train.value_counts().to_dict()
    },
    
    "time": {
        "baseline_seconds": round(baseline_time, 2),
        "search_minutes": round(search_time / 60, 2),
        "total_minutes": round((baseline_time + search_time) / 60, 2)
    },
    
    "baseline":  {
        "test_f1_macro": float(baseline_f1),
        "test_recall": float(baseline_recall),
        "test_precision":  float(baseline_precision),
        "test_accuracy": float(baseline_accuracy)
    },
    
    "tuned": {
        "cv_f1_macro": float(best_cv_score),
        "cv_f1_std": float(cv_results.loc[cv_results['rank_test_score'] == 1, 'std_test_score'].values[0]),
        "test_f1_macro": float(tuned_f1),
        "test_recall": float(tuned_recall),
        "test_precision": float(tuned_precision),
        "test_accuracy": float(tuned_accuracy),
        "best_params": {k: str(v) for k, v in best_params.items()}
    },
    
    "improvement": {
        "f1_macro_%": float((tuned_f1 - baseline_f1) * 100),
        "recall_%": float((tuned_recall - baseline_recall) * 100),
        "precision_%": float((tuned_precision - baseline_precision) * 100),
        "accuracy_%": float((tuned_accuracy - baseline_accuracy) * 100)
    },
    
    "top_5_combinations": [
        {
            "rank":  int(row['rank_test_score']),
            "cv_f1_mean": float(row['mean_test_score']),
            "cv_f1_std": float(row['std_test_score']),
            "params": row['params']
        }
        for _, row in top5.iterrows()
    ]
}

json_file = RESULTS_DIR / "random_search_results.json"
with open(json_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ {json_file}")

# CSV (tüm CV sonuçları)
csv_file = RESULTS_DIR / "random_search_cv_details.csv"
cv_results.to_csv(csv_file, index=False)
print(f"✓ {csv_file}")

# Text rapor
report_file = REPORTS_DIR / "random_search_report.txt"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("   HYPERPARAMETER TUNING RAPORU\n")
    f.write("=" * 70 + "\n\n")
    
    f.write(f"Tarih: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Method: RandomizedSearchCV ({N_ITER} iter, {N_FOLDS}-fold CV)\n\n")
    
    f.write("VERİ:\n")
    f.write("-" * 70 + "\n")
    f.write(f"Train:  {len(X_train):,} satır, {len(X_train.columns)} feature\n")
    f.write(f"Test:   {len(X_test):,} satır\n\n")
    
    f.write("BASELINE (default params):\n")
    f.write("-" * 70 + "\n")
    f.write(f"Test F1-macro:   {baseline_f1:.4f}\n")
    f.write(f"Test Recall:     {baseline_recall:.4f}\n")
    f.write(f"Test Precision: {baseline_precision:.4f}\n")
    f.write(f"Test Accuracy:  {baseline_accuracy:.4f}\n\n")
    
    f.write("TUNED (best params):\n")
    f.write("-" * 70 + "\n")
    f.write(f"CV F1-macro:    {best_cv_score:.4f}\n")
    f.write(f"Test F1-macro:  {tuned_f1:.4f}\n")
    f.write(f"Test Recall:     {tuned_recall:.4f}\n")
    f.write(f"Test Precision: {tuned_precision:.4f}\n")
    f.write(f"Test Accuracy:  {tuned_accuracy:.4f}\n\n")
    
    f.write("EN İYİ PARAMETRELER:\n")
    f.write("-" * 70 + "\n")
    for param, value in sorted(best_params.items()):
        f.write(f"  {param:20s}: {value}\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write(f"Toplam Süre: {(baseline_time + search_time)/60:.1f} dakika\n")
    f.write("=" * 70 + "\n")

print(f"✓ {report_file}")
print()

# ═══════════════════════════════════════════════════
# ÖZET
# ═══════════════════════════════════════════════════
print("=" * 70)
print("   ÖZET")
print("=" * 70)
print(f"✓ {N_ITER} kombinasyon × {N_FOLDS}-fold CV = {N_ITER * N_FOLDS} fit")
print(f"✓ Toplam süre: {(baseline_time + search_time)/60:.1f} dakika")
print(f"✓ Baseline Test F1:  {baseline_f1:.4f}")
print(f"✓ Tuned Test F1:     {tuned_f1:.4f} ({(tuned_f1-baseline_f1)*100:+.2f}%)")
print(f"\n📁 Kaydedilen dosyalar:")
print(f"  - {json_file}")
print(f"  - {csv_file}")
print(f"  - {report_file}")
print(f"  - {REPORTS_DIR}/random_search_*.png (3 görsel)")
print(f"\n🎯 Sonraki:  Final Model Eğitimi")
print("=" * 70)