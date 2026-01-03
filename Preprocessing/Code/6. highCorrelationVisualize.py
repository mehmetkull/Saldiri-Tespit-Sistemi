"""Redundant alan tespiti sonuçlarını görselleştir."""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "high_correlation_and_redundant_groups"
RESULTS_DIR = ROOT / "results" / "high_corr_visualize"


def main():
    print("Görseller oluşturuluyor...")
    
    # 1) Verileri yükle
    high_pairs = pd.read_csv(DATA_DIR / "high_corr_pairs.csv")
    groups = pd.read_csv(DATA_DIR / "redundant_groups.csv")

    # 2) Top 15 korelasyonlu çift
    fig, ax = plt.subplots(figsize=(10, 6))
    top = high_pairs.head(15).copy()
    top["pair"] = top["feature1"].str[:12] + " ↔ " + top["feature2"].str[:12]
    ax.barh(top["pair"], top["correlation"].abs(), color="#e74c3c", edgecolor="black")
    ax.set_xlabel("|Korelasyon|")
    ax.set_xlim(0.94, 1.01)
    ax.set_title("En Yüksek Korelasyonlu 15 Çift", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "top_corr_pairs.png", dpi=150)
    plt.close()
    
    # 3) Grup boyutları
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh([f"Grup {r['group_id']}" for _, r in groups.iterrows()], groups["size"], color="#f39c12", edgecolor="black")
    ax.set_xlabel("Feature Sayısı")
    ax.set_title("Redundant Grup Boyutları", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "group_sizes.png", dpi=150)
    plt.close()
    
    print(f"Tamamlandı.  Çıktılar:  {RESULTS_DIR}/")
    print("  - top_corr_pairs.png")
    print("  - group_sizes.png")


if __name__ == "__main__":
    main()