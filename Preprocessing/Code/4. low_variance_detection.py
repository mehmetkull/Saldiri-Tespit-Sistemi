"""Düşük varyanslı sütunları tespit eden modül"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Config
VAR_THRESHOLD = 1e-6
UNIQUE_CUT = 0.10
FREQ_CUT = 95

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "binary_labels.csv"
RESULTS_DIR = ROOT / "results" / "low_variance"


def main():
    print("\n" + "="*40)
    print("DÜŞÜK VARYANS TESPİT")
    print("="*40 + "\n")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1) Veri yükle
    print("[1/4] Veri yükleniyor...")
    df = pd.read_csv(DATA_PATH)
    
    label_col = " Label"
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != label_col]
    X = df[numeric_cols]
    
    print(f"   Shape: {df.shape} | Numerik:  {len(numeric_cols)}")
    
    # 2) Sabit sütunlar
    print("[2/4] Sabit sütunlar...")
    constant = [c for c in numeric_cols if X[c].nunique() <= 1]
    
    # 3) Düşük varyans (0 < var <= threshold)
    print("[3/4] Düşük varyans...")
    low_var = []
    for c in numeric_cols: 
        var = X[c]. var()
        if 0 < var <= VAR_THRESHOLD:
            low_var.append({"column": c, "variance": var})
    
    # 4) Near-zero variance
    print("[4/4] Near-zero variance...")
    nzv = []
    for c in numeric_cols:
        if c in constant:
            continue
        pct_unique = X[c].nunique() / len(X) * 100
        vc = X[c].value_counts()
        freq_ratio = vc. iloc[0] / vc. iloc[1] if len(vc) >= 2 else float("inf")
        
        if pct_unique <= UNIQUE_CUT and freq_ratio >= FREQ_CUT: 
            nzv.append({"column": c, "pct_unique": pct_unique, "freq_ratio": freq_ratio})
    
    # Karar
    to_drop = sorted(set(constant + [d["column"] for d in low_var] + [d["column"] for d in nzv]))
    
    # CSV çıktıları
    print("\nSonuçlar kaydediliyor...")
    pd.DataFrame({"column": constant}).to_csv(RESULTS_DIR / "constant_features.csv", index=False)
    pd.DataFrame(low_var).to_csv(RESULTS_DIR / "low_variance_numeric.csv", index=False)
    pd.DataFrame(nzv).to_csv(RESULTS_DIR / "near_zero_variance.csv", index=False)
    
    # MD rapor
    report = f"""# Düşük Varyans Raporu
**Tarih:** {datetime. now().strftime('%Y-%m-%d %H:%M:%S')}

## Özet
| Metrik | Değer |
|--------|-------|
| Toplam Feature | {len(numeric_cols)} |
| Sabit | {len(constant)} |
| Düşük Varyans | {len(low_var)} |
| Near-Zero Variance | {len(nzv)} |
| **Silinecek** | {len(to_drop)} |

## Sabit Sütunlar
{chr(10).join(f"- `{c}`" for c in constant) or "- Yok"}

## Düşük Varyans
{chr(10).join(f"- `{d['column']}` (var: {d['variance']:.2e})" for d in low_var) or "- Yok"}

## Near-Zero Variance
{chr(10).join(f"- `{d['column']}` (unique: {d['pct_unique']:.4f}%, freq: {d['freq_ratio']:.1f})" for d in nzv) or "- Yok"}

## Toplam Silinecekler
{chr(10).join(f"- `{c}`" for c in to_drop) or "- Yok"}

## Eşikler
- Varyans: < {VAR_THRESHOLD}
- Unique %: <= {UNIQUE_CUT}
- Freq ratio:  >= {FREQ_CUT}
"""
    (RESULTS_DIR / "variance_decision.md").write_text(report, encoding="utf-8")
    
    # Özet
    print(f"\n{'='*40}")
    print("ÖZET")
    print(f"{'='*40}")
    print(f"Toplam:     {len(numeric_cols)}")
    print(f"Sabit:       {len(constant)}")
    print(f"Düşük var:  {len(low_var)}")
    print(f"NZV:        {len(nzv)}")
    print(f"Silinecek:  {len(to_drop)}")
    print(f"{'='*40}")
    print(f"Çıktılar:   {RESULTS_DIR}/")


if __name__ == "__main__": 
    main()