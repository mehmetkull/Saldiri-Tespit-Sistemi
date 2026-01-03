import pandas as pd

df = pd.read_csv("variance_removed.csv")

# Dağılım kontrolü (önce)
print("Önce:", df[" Label"].value_counts(normalize=True).head())

# Shuffle 
# her seferinde aynı sonucu almak için random_state ayarlandı
# verileri karıştır ve indeksleri sıfırla
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Dağılım kontrolü (sonra)
print("Sonra:", df[" Label"]. value_counts(normalize=True).head())

# Kaydet
df.to_csv("shuffled.csv", index=False)
print("Kaydedildi.")