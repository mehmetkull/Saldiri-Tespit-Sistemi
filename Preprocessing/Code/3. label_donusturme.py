import pandas as pd
from pathlib import Path

""" Bu fonksiyon Label sütununu binary değerlere dönüştürür ve rapor üretir.
    Kural:
    - 'BENIGN' -> 0
    - Diğer tüm etiketler -> 1

    Not:
    - Label'i eksik olan satırlar silinir (NaN veya boş/boşluk).
    """

def donustur_label(input_file, output_file, report_file=None, label_col="Label"):
    # Veriyi oku
    df = pd.read_csv(input_file)
    original_rows = len(df)

    # Herhangi bir sütunda eksik değeri olan satır sayısı
    missing_any_rows = int(df.isna().any(axis=1).sum())

    # Label sütunu ismi doğrudan ' Label' olarak kullanılacak
    label_col = ' Label'

    # Label'i normalize ederek (NaN -> "") eksik/boş label satırlarını tespit et
    label_str = df[label_col].fillna("").astype(str).str.strip()
    missing_label_rows = int(label_str.eq("").sum())
    # label'i eksik satırları sil
    df = df.loc[~label_str.eq("")].copy()
    dropped_rows = original_rows - len(df)

    # Silme sonrası orijinal label dağılımını raporlamak için sakla
    original_label_counts = df[label_col].value_counts()

    # 'BENIGN' tespiti için büyük-küçük harf ve boşlukları normalize et
    label_norm = df[label_col].astype(str).str.strip().str.upper()
    benign_mask = label_norm.eq("BENIGN")
    benign_count = int(benign_mask.sum())
    attack_count = int((~benign_mask).sum())

    # Binary dönüşüm: BENIGN=0, diğerleri=1
    df[label_col] = (~benign_mask).astype(int)
    binary_counts = df[label_col].value_counts()

    # Dönüşen veriyi kaydet
    df.to_csv(output_file, index=False)

    if report_file is None:
        # Rapor yolu verilmezse çıktının yanına otomatik isimle yaz
        out_path = Path(output_file)
        report_file = str(out_path.with_name(out_path.stem + "_label_donusturme_raporu.txt"))

    report_lines = []
    report_lines.append("LABEL DÖNÜŞTÜRME RAPORU")
    report_lines.append("=" * 60)
    report_lines.append(f"Girdi: {input_file}")
    report_lines.append(f"Çıktı: {output_file}")
    report_lines.append("")

    report_lines.append("1) Satır Sayıları")
    report_lines.append(f"- Orijinal satır: {original_rows}")
    report_lines.append(f"- Herhangi bir sütunda eksik olan satır: {missing_any_rows}")
    report_lines.append(f"- Label eksik olan satır: {missing_label_rows}")
    report_lines.append(f"- Label eksik olduğu için silinen: {dropped_rows}")
    report_lines.append(f"- Kalan satır: {len(df)}")
    report_lines.append("")

    report_lines.append("2) Dönüşüm Kuralı")
    report_lines.append("- 'BENIGN' -> 0")
    report_lines.append("- Diğerleri -> 1")
    report_lines.append("")

    report_lines.append("3) Orijinal Label Dağılımı (silme sonrası)")
    for label_value, count in original_label_counts.items():
        report_lines.append(f"- {label_value}: {count}")
    report_lines.append("")

    report_lines.append("4) Dönüşüm Sonrası (Binary) Dağılım")
    report_lines.append(f"- 0 (BENIGN): {binary_counts.get(0, 0)}")
    report_lines.append(f"- 1 (Saldırı): {binary_counts.get(1, 0)}")
    report_lines.append(f"- BENIGN olarak sayılan: {benign_count}")
    report_lines.append(f"- Saldırı olarak sayılan: {attack_count}")

    if len(df) > 0:
        report_lines.append("")
        report_lines.append("5) Yüzdeler")
        report_lines.append(f"- 0 (BENIGN): %{binary_counts.get(0, 0) / len(df) * 100:.2f}")
        report_lines.append(f"- 1 (Saldırı): %{binary_counts.get(1, 0) / len(df) * 100:.2f}")

    # Raporu yaz
    Path(report_file).parent.mkdir(parents=True, exist_ok=True)
    Path(report_file).write_text("\n".join(report_lines), encoding="utf-8")

    return df

if __name__ == "__main__":
    try:
        # Kullanım örneği
        input_file = "c:/Users/kul38/Desktop/IDS Project/training/cleaned_data.csv"
        output_file = "c:/Users/kul38/Desktop/IDS Project/training/binary_labels.csv"
        report_file = "c:/Users/kul38/Desktop/IDS Project/training/label_donusturme_raporu.txt"
        
        df = donustur_label(input_file, output_file, report_file=report_file)

        print(f"CSV kaydedildi: {output_file}")
        print(f"Rapor kaydedildi: {report_file}")

    except Exception as e:
        print(f"HATA: {e}")
