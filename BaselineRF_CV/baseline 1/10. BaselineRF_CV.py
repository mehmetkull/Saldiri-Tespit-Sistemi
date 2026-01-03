"""
Baseline RF CV + Multi-Metric Feature Importance
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score
from imblearn.pipeline import Pipeline
from imblearn. over_sampling import SMOTE

# ═══════════════════════════════════════════════════
# AYARLAR
# ═══════════════════════════════════════════════════
TRAIN_FILE = "train.csv"
RESULTS_DIR = Path("results")
REPORTS_DIR = Path("reports")
TARGET_COLUMN = " Label"  # Boşluklu
RANDOM_STATE = 42
N_FOLDS = 5

RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

WEIGHTS = {'mdi':  0.40, 'f1': 0.30, 'recall': 0.20, 'precision': 0.10}

RESULTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════
# VERİ YÜKLEME + DEBUG
# ═══════════════════════════════════════════════════
print("=" * 70)
print("   BASELINE RF CV + FEATURE IMPORTANCE")
print("=" * 70)

df_train = pd.read_csv(TRAIN_FILE)

# DEBUG: Sütunları kontrol et
print(f"\n🔍 DEBUG: Sütun sayısı: {len(df_train. columns)}")
print(f"🔍 Label içeren sütunlar: {[col for col in df_train. columns if 'Label' in col]}")

# Label sütununu düzgün al
if TARGET_COLUMN not in df_train.columns:
    # Alternatif: boşluksuz dene
    if 'Label' in df_train.columns:
        TARGET_COLUMN = 'Label'
        print(f"⚠️  ' Label' bulunamadı, 'Label' kullanılıyor")
    else:
        print(f"❌ HATA: Label sütunu bulunamadı!")
        print(f"Mevcut sütunlar: {df_train.columns. tolist()}")
        exit(1)

# X ve y ayır
X_train = df_train.drop(columns=[TARGET_COLUMN])
y_train = df_train[TARGET_COLUMN]

# y_train'i kesinlikle 1D yap
if isinstance(y_train, pd.DataFrame):
    y_train = y_train.squeeze()
if isinstance(y_train, pd.DataFrame):
    y_train = y_train.iloc[:, 0]
# Numpy array'e çevirip reshape
y_train = pd.Series(np.ravel(y_train), name=TARGET_COLUMN)

print(f"\n✓ X_train shape: {X_train.shape}")
print(f"✓ y_train shape:  {y_train.shape}")
print(f"✓ y_train type: {type(y_train)}")
print(f"✓ y_train ilk 3 değer: {y_train. head(3).tolist()}")

# y_train 1D değilse çık
if len(y_train.shape) != 1:
    print(f"\n❌ HATA:   y_train hala {y_train.shape} boyutunda!")
    exit(1)

print(f"✓ {N_FOLDS}-fold CV, Random State: {RANDOM_STATE}\n")

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
feature_names = X_train.columns.tolist()
n_features = len(feature_names)

# ═══════════════════════════════════════════════════
# CV + IMPORTANCE
# ═══════════════════════════════════════════════════
def run_cv_with_importance(pipeline, X, y, label):
    """CV yap, metrikler + importance hesapla"""
    
    scores = {'f1': [], 'recall': [], 'precision': [], 'roc_auc': [], 'acc': []}
    imp_mdi, imp_f1, imp_recall, imp_precision = [], [], [], []
    
    print(f"⏳ {label} CV...")
    start = time.time()
    
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
        # iloc kullan (daha güvenli)
        X_tr = X.iloc[tr_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_tr = y.iloc[tr_idx].copy()
        y_val = y.iloc[val_idx].copy()
        
        # y_tr ve y_val'i kesinlikle 1D numpy array yap
        y_tr = np.ravel(y_tr)
        y_val = np.ravel(y_val)
        
        pipeline.fit(X_tr, y_tr)
        rf = pipeline.named_steps['clf']
        
        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)
        
        # Metrikler
        f1_base = f1_score(y_val, y_pred, average='macro')
        rec_base = recall_score(y_val, y_pred, average='macro')
        prec_base = precision_score(y_val, y_pred, average='macro')
        
        scores['f1'].append(f1_base)
        scores['recall'].append(rec_base)
        scores['precision'].append(prec_base)
        
        # Binary classification için y_proba[:,1] kullan (pozitif sınıf olasılığı)
        if y_proba.shape[1] == 2:
            scores['roc_auc'].append(roc_auc_score(y_val, y_proba[:, 1]))
        else:
            scores['roc_auc'].append(roc_auc_score(y_val, y_proba, average='macro', multi_class='ovr'))
        
        scores['acc'].append(accuracy_score(y_val, y_pred))
        
        # MDI importance
        imp_mdi.append(rf.feature_importances_)
        
        # Permutation importance
        f1_drops, rec_drops, prec_drops = [], [], []
        
        for i in range(n_features):
            X_perm = X_val.copy()
            X_perm. iloc[:, i] = np.random.permutation(X_perm.iloc[:, i]. values)
            y_perm = pipeline.predict(X_perm)
            
            f1_drops.append(max(0, f1_base - f1_score(y_val, y_perm, average='macro')))
            rec_drops.append(max(0, rec_base - recall_score(y_val, y_perm, average='macro')))
            prec_drops.append(max(0, prec_base - precision_score(y_val, y_perm, average='macro')))
        
        imp_f1.append(f1_drops)
        imp_recall.append(rec_drops)
        imp_precision.append(prec_drops)
        
        print(f"  Fold {fold}: F1={f1_base:.3f} Recall={rec_base:.3f} Precision={prec_base:.3f}")
    
    elapsed = time.time() - start
    print(f"✓ {elapsed:.1f}s\n")
    
    # Ortalama importance
    mdi = np.mean(imp_mdi, axis=0)
    f1_drop = np.mean(imp_f1, axis=0)
    rec_drop = np.mean(imp_recall, axis=0)
    prec_drop = np.mean(imp_precision, axis=0)
    
    # Normalize + ağırlıklandır
    def norm(arr):
        s = arr.sum()
        return arr / s if s > 0 else arr
    
    combined = (WEIGHTS['mdi'] * norm(mdi) + 
                WEIGHTS['f1'] * norm(f1_drop) +
                WEIGHTS['recall'] * norm(rec_drop) +
                WEIGHTS['precision'] * norm(prec_drop))
    
    importance = {
        'mdi': {'mean': mdi, 'std': np.std(imp_mdi, axis=0)},
        'f1': {'mean': f1_drop, 'std': np. std(imp_f1, axis=0)},
        'recall': {'mean': rec_drop, 'std': np.std(imp_recall, axis=0)},
        'precision': {'mean': prec_drop, 'std': np.std(imp_precision, axis=0)},
        'combined': combined
    }
    
    return scores, importance, elapsed

# ═══════════════════════════════════════════════════
# BASELINE 1: SMOTE'suz
# ═══════════════════════════════════════════════════
print("=" * 70)
print("   BASELINE 1: SMOTE'suz")
print("=" * 70)

pipeline_no_smote = Pipeline([('clf', RandomForestClassifier(**RF_PARAMS))])
scores_no, imp_no, time_no = run_cv_with_importance(pipeline_no_smote, X_train, y_train, "SMOTE'suz")

print("📊 SONUÇLAR")
for k, v in scores_no.items():
    print(f"  {k:12s}: {np.mean(v):.3f} ± {np.std(v):.3f}")
print()

# ═══════════════════════════════════════════════════
# BASELINE 2: SMOTE'lu
# ═══════════════════════════════════════════════════
print("=" * 70)
print("   BASELINE 2: SMOTE'lu")
print("=" * 70)

pipeline_smote = Pipeline([
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('clf', RandomForestClassifier(**RF_PARAMS))
])
scores_smote, imp_smote, time_smote = run_cv_with_importance(pipeline_smote, X_train, y_train, "SMOTE'lu")

print("📊 SONUÇLAR")
for k, v in scores_smote.items():
    print(f"  {k:12s}: {np.mean(v):.3f} ± {np.std(v):.3f}")
print()

# ═══════════════════════════════════════════════════
# EN İYİ PIPELINE SEÇ
# ═══════════════════════════════════════════════════
f1_no = np.mean(scores_no['f1'])
f1_smote = np.mean(scores_smote['f1'])
f1_improvement = ((f1_smote - f1_no) / f1_no) * 100

best = "smote" if f1_improvement > 2 else "no_smote"
best_imp = imp_smote if best == "smote" else imp_no

print("=" * 70)
print(f"   EN İYİ: {best.upper()} (F1: {f1_smote if best=='smote' else f1_no:.3f}, İyileşme: {f1_improvement:+.1f}%)")
print("=" * 70)
print()

# ═══════════════════════════════════════════════════
# IMPORTANCE DATAFRAME
# ═══════════════════════════════════════════════════
df_imp = pd.DataFrame({
    'feature': feature_names,
    'mdi_mean': best_imp['mdi']['mean'],
    'mdi_std':  best_imp['mdi']['std'],
    'f1_mean': best_imp['f1']['mean'],
    'f1_std':  best_imp['f1']['std'],
    'recall_mean':  best_imp['recall']['mean'],
    'recall_std': best_imp['recall']['std'],
    'precision_mean': best_imp['precision']['mean'],
    'precision_std': best_imp['precision']['std'],
    'combined':  best_imp['combined']
}).sort_values('combined', ascending=False).reset_index(drop=True)

df_imp['rank'] = range(1, len(df_imp) + 1)
df_imp['cumulative'] = df_imp['combined'].cumsum()

feat_90 = len(df_imp[df_imp['cumulative'] <= 0.90])
feat_95 = len(df_imp[df_imp['cumulative'] <= 0.95])

# ═══════════════════════════════════════════════════
# TÜM ÖZELLİKLER RAPORU
# ═══════════════════════════════════════════════════
print("📋 TÜM ÖZELLİKLER (Combined Importance)")
print("-" * 110)
print(f"{'Rank':<5} {'Feature':<35} {'Combined':<11} {'MDI':<11} {'F1 Drop':<11} {'Recall':<11} {'Precision':<11}")
print("-" * 110)

for _, row in df_imp.iterrows():
    print(f"{row['rank']:<5} {row['feature']:<35} "
          f"{row['combined']:.6f}   "
          f"{row['mdi_mean']:.6f}   "
          f"{row['f1_mean']:.6f}   "
          f"{row['recall_mean']:.6f}   "
          f"{row['precision_mean']:.6f}")

print("-" * 110)
print(f"Toplam: {n_features} feature")
print(f"Kümülatif %90: {feat_90} feature")
print(f"Kümülatif %95: {feat_95} feature\n")

# ═══════════════════════════════════════════════════
# GÖRSELLEŞTİRME
# ═══════════════════════════════════════════════════
print("📊 Görseller oluşturuluyor...")

sns.set_style("whitegrid")

# 1. Top 30 Bar Chart
fig, ax = plt.subplots(figsize=(10, 12))
top30 = df_imp.head(30)
ax.barh(range(len(top30)), top30['combined'], color='steelblue')
ax.set_yticks(range(len(top30)))
ax.set_yticklabels(top30['feature'], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Combined Importance', fontsize=11, fontweight='bold')
ax.set_title(f'Top 30 Features - Combined Importance ({best.upper()})', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "importance_top30.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Cumulative
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(1, n_features+1), df_imp['cumulative']*100, linewidth=2, color='steelblue')
ax.axhline(y=90, color='red', linestyle='--', linewidth=2, label='90%')
ax.axhline(y=95, color='orange', linestyle='--', linewidth=2, label='95%')
ax.axvline(x=feat_90, color='red', linestyle=':', alpha=0.5)
ax.axvline(x=feat_95, color='orange', linestyle=':', alpha=0.5)
ax.set_xlabel('Number of Features', fontsize=11, fontweight='bold')
ax.set_ylabel('Cumulative Importance (%)', fontsize=11, fontweight='bold')
ax.set_title('Cumulative Feature Importance', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.text(feat_90, 88, f'{feat_90}', ha='center', fontsize=9, color='red')
ax.text(feat_95, 93, f'{feat_95}', ha='center', fontsize=9, color='orange')
plt.tight_layout()
plt.savefig(REPORTS_DIR / "importance_cumulative.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Heatmap (Top 40)
fig, ax = plt.subplots(figsize=(10, 14))
top40 = df_imp.head(40)
hmap = top40[['mdi_mean', 'f1_mean', 'recall_mean', 'precision_mean']].T
hmap_norm = hmap.div(hmap.max(axis=1), axis=0)
sns.heatmap(hmap_norm, xticklabels=top40['feature'], 
            yticklabels=['MDI', 'F1 Drop', 'Recall Drop', 'Precision Drop'],
            cmap='YlOrRd', cbar_kws={'label': 'Normalized'}, ax=ax)
ax.set_title('Top 40 Features - Multi-Metric Heatmap', fontsize=13, fontweight='bold')
plt.xticks(rotation=90, fontsize=8)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "importance_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

print("✓ 3 görsel kaydedildi\n")

# ═══════════════════════════════════════════════════
# KAYDET
# ═══════════════════════════════════════════════════
print("💾 Sonuçlar kaydediliyor...")

df_imp.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

results = {
    "train_file": TRAIN_FILE,
    "n_samples": len(X_train),
    "n_features": n_features,
    "n_folds": N_FOLDS,
    "random_state": RANDOM_STATE,
    "importance_weights": WEIGHTS,
    "no_smote": {
        "cv_time": time_no,
        "metrics": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in scores_no.items()}
    },
    "smote": {
        "cv_time": time_smote,
        "metrics": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in scores_smote.items()}
    },
    "recommended": best,
    "f1_improvement_%": float(f1_improvement),
    "cumulative_90%": int(feat_90),
    "cumulative_95%": int(feat_95),
    "all_features": df_imp[['rank', 'feature', 'combined']].to_dict('records')
}

with open(RESULTS_DIR / "baseline_cv.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ {RESULTS_DIR}/feature_importance.csv")
print(f"✓ {RESULTS_DIR}/baseline_cv.json")
print(f"✓ {REPORTS_DIR}/ (3 görsel)")

print("\n" + "=" * 70)
print("   ÖZET")
print("=" * 70)
print(f"✓ {n_features} feature, 4 metrikle ağırlıklandırıldı")
print(f"✓ En iyi: {best.upper()}")
print(f"✓ F1-macro: {f1_smote if best=='smote' else f1_no:.3f}")
print(f"✓ %90 için {feat_90} feature, %95 için {feat_95} feature")
print(f"\n🎯 Sonraki:  Korelasyon Analizi + Feature Selection")
print("=" * 70)