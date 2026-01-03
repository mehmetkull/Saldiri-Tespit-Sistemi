"""
Baseline RF CV + MDI Feature Importance + Destination Port Dependency Test
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
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ═══════════════════════════════════════════════════
# AYARLAR
# ═══════════════════════════════════════════════════
TRAIN_FILE = "/home/azureuser/cloudfiles/code/IDS_Project/train.csv"
RESULTS_DIR = Path("/home/azureuser/cloudfiles/code/IDS_Project/results")
REPORTS_DIR = Path("/home/azureuser/cloudfiles/code/IDS_Project/reports")
TARGET_COLUMN = " Label"  # Boşluklu
RANDOM_STATE = 42
N_FOLDS = 5

RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 50,
    'min_samples_leaf': 20,
    'max_features': 0.2,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

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
# CV + MDI & DESTINATION PORT PERMUTATION
# ═══════════════════════════════════════════════════
def run_cv_with_importance(clf, X, y):
    """CV yap, MDI hesapla + SADECE Destination Port için permutation importance hesapla"""
    
    scores = {'f1': [], 'recall': [], 'precision': [], 'roc_auc': [], 'acc': []}
    imp_mdi = []
    dest_port_f1_drops, dest_port_rec_drops, dest_port_prec_drops = [], [], []
    
    # Destination Port feature'ı bul
    dest_port_col = None
    dest_port_idx = None
    for i, col in enumerate(X.columns):
        if 'destination' in col.lower() and 'port' in col.lower():
            dest_port_col = col
            dest_port_idx = i
            break
    
    print(f"⏳ CV başlıyor...")
    if dest_port_col:
        print(f"✓ Destination Port sütunu bulundu: '{dest_port_col}' (index: {dest_port_idx})")
        print(f"ℹ️  Bu feature için permutation importance hesaplanacak (model bağımlılık analizi)")
    else:
        print(f"⚠️  Destination Port sütunu bulunamadı - permutation importance hesaplanmayacak")
    
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
        
        clf.fit(X_tr, y_tr)
        
        y_pred = clf.predict(X_val)
        y_proba = clf.predict_proba(X_val)
        
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
        
        # MDI importance (tüm feature'lar için)
        imp_mdi.append(clf.feature_importances_)
        
        # SADECE Destination Port için permutation importance
        if dest_port_col and dest_port_idx is not None:
            X_perm = X_val.copy()
            X_perm.iloc[:, dest_port_idx] = np.random.permutation(X_perm.iloc[:, dest_port_idx].values)
            y_perm = clf.predict(X_perm)
            
            f1_drop = max(0, f1_base - f1_score(y_val, y_perm, average='macro'))
            rec_drop = max(0, rec_base - recall_score(y_val, y_perm, average='macro'))
            prec_drop = max(0, prec_base - precision_score(y_val, y_perm, average='macro'))
            
            dest_port_f1_drops.append(f1_drop)
            dest_port_rec_drops.append(rec_drop)
            dest_port_prec_drops.append(prec_drop)
        
        print(f"  Fold {fold}: F1={f1_base:.3f} Recall={rec_base:.3f} Precision={prec_base:.3f}")
    
    elapsed = time.time() - start
    print(f"✓ {elapsed:.1f}s\n")
    
    # Ortalama importance
    mdi = np.mean(imp_mdi, axis=0)
    
    # Combined importance: MDI tek başına kullanılıyor 
    def norm(arr):
        s = arr.sum()
        return arr / s if s > 0 else arr
    
    combined = norm(mdi)  # MDI tek başına kullanılıyor
    
    importance = {
        'mdi': {'mean': mdi, 'std': np.std(imp_mdi, axis=0)},
        'combined': combined
    }
    
    # Destination Port permutation importance ekle (eğer hesaplandıysa)
    if dest_port_col and len(dest_port_f1_drops) > 0:
        importance['dest_port_perm'] = {
            'f1_drop': {'mean': np.mean(dest_port_f1_drops), 'std': np.std(dest_port_f1_drops)},
            'recall_drop': {'mean': np.mean(dest_port_rec_drops), 'std': np.std(dest_port_rec_drops)},
            'precision_drop': {'mean': np.mean(dest_port_prec_drops), 'std': np.std(dest_port_prec_drops)},
            'feature': dest_port_col,
            'index': dest_port_idx
        }
    
    return scores, importance, elapsed

# ═══════════════════════════════════════════════════
# BASELINE RANDOM FOREST
# ═══════════════════════════════════════════════════
print("=" * 70)
print("   BASELINE RANDOM FOREST (SMOTE'suz)")
print("=" * 70)

clf = RandomForestClassifier(**RF_PARAMS)
scores, importance, elapsed_time = run_cv_with_importance(clf, X_train, y_train)

print("📊 SONUÇLAR")
for k, v in scores.items():
    print(f"  {k:12s}: {np.mean(v):.3f} ± {np.std(v):.3f}")
print()

# ═══════════════════════════════════════════════════
# IMPORTANCE DATAFRAME
# ═══════════════════════════════════════════════════
df_imp = pd.DataFrame({
    'feature': feature_names,
    'mdi_mean': importance['mdi']['mean'],
    'mdi_std':  importance['mdi']['std'],
    'importance':  importance['combined']
}).sort_values('importance', ascending=False).reset_index(drop=True)

df_imp['rank'] = range(1, len(df_imp) + 1)
df_imp['cumulative'] = df_imp['importance'].cumsum()

feat_90 = len(df_imp[df_imp['cumulative'] <= 0.90])
feat_95 = len(df_imp[df_imp['cumulative'] <= 0.95])

# ═══════════════════════════════════════════════════
# TÜM ÖZELLİKLER RAPORU
# ═══════════════════════════════════════════════════
print("📋 TÜM ÖZELLİKLER (MDI-Based Importance)")
print("-" * 80)
print(f"{'Rank':<5} {'Feature':<40} {'MDI':<12} {'Importance':<12}")
print("-" * 80)

for _, row in df_imp.iterrows():
    print(f"{row['rank']:<5} {row['feature']:<40} "
          f"{row['mdi_mean']:.6f}   "
          f"{row['importance']:.6f}")

print("-" * 80)
print(f"Toplam: {n_features} feature")
print(f"Kümülatif %90: {feat_90} feature")
print(f"Kümülatif %95: {feat_95} feature")

# Destination Port Permutation Importance Raporu
if 'dest_port_perm' in importance:
    dp_info = importance['dest_port_perm']
    print("\n" + "=" * 80)
    print("   DESTINATION PORT BAĞIMLILIK ANALİZİ (Permutation Importance)")
    print("=" * 80)
    print(f"Feature: {dp_info['feature']}")
    print(f"Index: {dp_info['index']}")
    print(f"\nModel Performans Düşüşü (Destination Port karıştırıldığında):")
    print(f"  F1 Drop:       {dp_info['f1_drop']['mean']:.4f} ± {dp_info['f1_drop']['std']:.4f}")
    print(f"  Recall Drop:   {dp_info['recall_drop']['mean']:.4f} ± {dp_info['recall_drop']['std']:.4f}")
    print(f"  Precision Drop: {dp_info['precision_drop']['mean']:.4f} ± {dp_info['precision_drop']['std']:.4f}")
    
    avg_drop = (dp_info['f1_drop']['mean'] + dp_info['recall_drop']['mean'] + dp_info['precision_drop']['mean']) / 3
    print(f"\n  Ortalama Drop: {avg_drop:.4f}")
    
    if avg_drop > 0.05:
        print(f"\n⚠️  YÜKSEK BAĞIMLILIK: Model Destination Port'a çok bağımlı!")
    elif avg_drop > 0.02:
        print(f"\nℹ️  ORTA BAĞIMLILIK: Model Destination Port'u kullanıyor.")
    else:
        print(f"\n✓ DÜŞÜK BAĞIMLILIK: Model Destination Port'a fazla bağımlı değil.")
    print("=" * 80)
print()

# ═══════════════════════════════════════════════════
# GÖRSELLEŞTİRME
# ═══════════════════════════════════════════════════
print("📊 Görseller oluşturuluyor...")

sns.set_style("whitegrid")

# 1. Top 30 Bar Chart
fig, ax = plt.subplots(figsize=(10, 12))
top30 = df_imp.head(30)
ax.barh(range(len(top30)), top30['importance'], color='steelblue')
ax.set_yticks(range(len(top30)))
ax.set_yticklabels(top30['feature'], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance (MDI)', fontsize=11, fontweight='bold')
ax.set_title('Top 30 Features - MDI Importance', fontsize=13, fontweight='bold')
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

# 3. MDI Heatmap (Top 40)
fig, ax = plt.subplots(figsize=(10, 8))
top40 = df_imp.head(40)
data_for_heatmap = top40[['importance']].T
sns.heatmap(data_for_heatmap, xticklabels=top40['feature'], 
            yticklabels=['MDI Importance'],
            cmap='YlOrRd', cbar_kws={'label': 'Importance'}, ax=ax, annot=False)
ax.set_title('Top 40 Features - MDI Importance', fontsize=13, fontweight='bold')
plt.xticks(rotation=90, fontsize=8)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "importance_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Destination Port Dependency Visualization (eğer hesaplandıysa)
if 'dest_port_perm' in importance:
    dp_info = importance['dest_port_perm']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics = ['F1', 'Recall', 'Precision']
    drops = [
        dp_info['f1_drop']['mean'],
        dp_info['recall_drop']['mean'],
        dp_info['precision_drop']['mean']
    ]
    stds = [
        dp_info['f1_drop']['std'],
        dp_info['recall_drop']['std'],
        dp_info['precision_drop']['std']
    ]
    
    colors = ['#e74c3c' if d > 0.05 else '#f39c12' if d > 0.02 else '#27ae60' for d in drops]
    bars = ax.bar(metrics, drops, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Performance Drop', fontsize=11, fontweight='bold')
    ax.set_title(f'Model Dependency on {dp_info["feature"]}\n(Permutation Importance)', 
                 fontsize=13, fontweight='bold')
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1, alpha=0.5, label='High Dependency')
    ax.axhline(y=0.02, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Medium Dependency')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, drop, std) in enumerate(zip(bars, drops, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.005,
                f'{drop:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "dest_port_dependency.png", dpi=300, bbox_inches='tight')
    plt.close()
    num_visuals = 5
else:
    num_visuals = 4

# 5. Confusion Matrix (Final modeli tüm train datası ile eğit)
print(f"\n📊 Confusion Matrix oluşturuluyor...")
clf_final = RandomForestClassifier(**RF_PARAMS)
clf_final.fit(X_train, y_train)
y_pred_final = clf_final.predict(X_train)

cm = confusion_matrix(y_train, y_pred_final)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('Confusion Matrix - Training Data', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ {num_visuals} görsel kaydedildi\n")

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
    "importance_method": "MDI (Mean Decrease Impurity)",
    "metrics": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in scores.items()},
    "cv_time": elapsed_time,
    "cumulative_90%": int(feat_90),
    "cumulative_95%": int(feat_95),
    "all_features": df_imp[['rank', 'feature', 'importance']].to_dict('records')
}

# Destination Port bilgilerini ekle
if 'dest_port_perm' in importance:
    dp_info = importance['dest_port_perm']
    results['destination_port_analysis'] = {
        "feature": dp_info['feature'],
        "index": int(dp_info['index']),
        "permutation_importance": {
            "f1_drop": {"mean": float(dp_info['f1_drop']['mean']), "std": float(dp_info['f1_drop']['std'])},
            "recall_drop": {"mean": float(dp_info['recall_drop']['mean']), "std": float(dp_info['recall_drop']['std'])},
            "precision_drop": {"mean": float(dp_info['precision_drop']['mean']), "std": float(dp_info['precision_drop']['std'])}
        },
        "dependency_level": "high" if (dp_info['f1_drop']['mean'] + dp_info['recall_drop']['mean'] + dp_info['precision_drop']['mean'])/3 > 0.05 
                            else "medium" if (dp_info['f1_drop']['mean'] + dp_info['recall_drop']['mean'] + dp_info['precision_drop']['mean'])/3 > 0.02 
                            else "low"
    }

with open(RESULTS_DIR / "baseline_cv.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ {RESULTS_DIR}/feature_importance.csv")
print(f"✓ {RESULTS_DIR}/baseline_cv.json")
if 'dest_port_perm' in importance:
    print(f"✓ {REPORTS_DIR}/ (5 görsel: top30, cumulative, heatmap, dest_port_dependency, confusion_matrix)")
else:
    print(f"✓ {REPORTS_DIR}/ (4 görsel: top30, cumulative, heatmap, confusion_matrix)")

print("\n" + "=" * 70)
print("   ÖZET")
print("=" * 70)
print(f"✓ {n_features} feature")
print(f"✓ Feature Importance: MDI (Mean Decrease Impurity) - CIC-IDS 2017 standart")
if 'dest_port_perm' in importance:
    dp_info = importance['dest_port_perm']
    avg_drop = (dp_info['f1_drop']['mean'] + dp_info['recall_drop']['mean'] + dp_info['precision_drop']['mean']) / 3
    print(f"✓ Destination Port Bağımlılık: {avg_drop:.4f} (Permutation Test)")
print(f"✓ F1-macro: {np.mean(scores['f1']):.3f} ± {np.std(scores['f1']):.3f}")
print(f"✓ %90 için {feat_90} feature, %95 için {feat_95} feature")
print(f"\n🎯 Sonraki: Korelasyon Analizi + Feature Selection")
print("=" * 70)