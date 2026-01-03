"""Numerik feature'larda redundant alan tespiti (yüksek korelasyon + MI).

 |r|>0.95 korelasyon + redundant gruplar + MI ile temsilci seçimi
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.feature_selection import mutual_info_classif

# Config
CORR_THRESHOLD = 0.95
RANDOM_STATE = 42


# Paths
ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "binary_labels.csv"
RESULTS_DIR = ROOT / "results" / "high_correlation_and_redundant_groups"



def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Veri yükle
    print("[1/5] Veri yükleniyor...")
    df = pd.read_csv(INPUT_CSV)
    
    label_col = " Label"
    X = df.drop(columns=[label_col])
    y = df[label_col].astype("category").cat.codes.to_numpy()
    features = X.columns. tolist()
    
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"   Shape: {X.shape} | Label: {label_col}")

    # 2) Korelasyon + yüksek korelasyon çiftleri
    print("[2/5] Korelasyon hesaplanıyor...")
    corr = X.corr(method="pearson")
    
    pairs = []
    for i, f1 in enumerate(features):
        for f2 in features[i+1:]: 
            r = corr.loc[f1, f2]
            if abs(r) > CORR_THRESHOLD:
                pairs. append({"feature1": f1, "feature2": f2, "correlation": r})
    
    high_pairs = pd.DataFrame(pairs).sort_values("correlation", key=abs, ascending=False)
    high_pairs.to_csv(RESULTS_DIR / "high_corr_pairs.csv", index=False)
    print(f"   Yüksek korr çift: {len(pairs)}")

    # 3) Redundant gruplar
    print("[3/5] Redundant gruplar bulunuyor...")
    G = nx.Graph()
    G.add_nodes_from(features)
    for _, row in high_pairs.iterrows():
        G.add_edge(row["feature1"], row["feature2"])
    
    groups = [sorted(c) for c in nx.connected_components(G) if len(c) > 1]
    groups. sort(key=lambda g: (-len(g), g[0]))
    
    group_features = list({f for grp in groups for f in grp})
    print(f"   Grup sayısı: {len(groups)} | Grup içi feature:  {len(group_features)}")

    # 4) MI hesapla (TÜM VERİ)
    print("[4/5] MI hesaplanıyor (tüm veri, birkaç dakika sürebilir)...")
    
    mi_scores = {}
    if group_features:
        mi = mutual_info_classif(X[group_features], y, random_state=RANDOM_STATE, n_jobs=-1)
        mi_scores = dict(zip(group_features, mi))

    # 5) Temsilci seç + raporla
    print("[5/5] Raporlar yazılıyor...")
    
    to_drop = []
    group_records = []
    
    for gid, grp in enumerate(groups, 1):
        rep = max(grp, key=lambda f: mi_scores. get(f, 0))
        dropped = [f for f in grp if f != rep]
        to_drop.extend(dropped)
        
        group_records. append({
            "group_id": gid,
            "size": len(grp),
            "representative": rep,
            "rep_mi": round(mi_scores.get(rep, 0), 4),
            "dropped": ";".join(dropped)
        })
    
    kept = [f for f in features if f not in to_drop]

    # Çıktılar
    pd.DataFrame(group_records).to_csv(RESULTS_DIR / "redundant_groups.csv", index=False)
    pd.DataFrame({"feature": to_drop}).to_csv(RESULTS_DIR / "dropped_by_correlation.csv", index=False)
    pd.DataFrame({"feature": kept}).to_csv(RESULTS_DIR / "kept_features.csv", index=False)
    
    stats = {
        "input_features": len(features),
        "high_corr_pairs": len(pairs),
        "redundant_groups":  len(groups),
        "dropped": len(to_drop),
        "kept":  len(kept),
        "corr_threshold": CORR_THRESHOLD
    }
    (RESULTS_DIR / "correlation_stats.json").write_text(json.dumps(stats, indent=2))

    # Özet
    print(f"\n{'='*40}")
    print("ÖZET")
    print(f"{'='*40}")
    print(f"Giriş feature:      {len(features)}")
    print(f"Yüksek korr çift:   {len(pairs)}")
    print(f"Redundant grup:     {len(groups)}")
    print(f"Silinen:             {len(to_drop)}")
    print(f"Kalan:              {len(kept)}")
    print(f"{'='*40}")
    print(f"Çıktılar:  {RESULTS_DIR}/")


if __name__ == "__main__": 
    main()