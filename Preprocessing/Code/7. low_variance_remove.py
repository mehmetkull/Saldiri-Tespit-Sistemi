import pandas as pd
from pathlib import Path


DROP_COLS = [
	" Bwd Avg Bytes/Bulk",
	" Bwd Avg Packets/Bulk",
	" Bwd PSH Flags",
	" Bwd URG Flags",
	" CWE Flag Count",
	" ECE Flag Count",
	" Fwd Avg Bulk Rate",
	" Fwd Avg Packets/Bulk",
	" Fwd URG Flags",
	" RST Flag Count",
	"Bwd Avg Bulk Rate",
	"Fwd Avg Bytes/Bulk",
]


def main() -> None:
	src = Path("binary_labels.csv")
	dst = Path("variance_removed.csv")

	if not src.exists():
		raise FileNotFoundError(f"Girdi dosyası bulunamadı: {src}")

	header_df = pd.read_csv(src, nrows=0)
	cols = list(header_df.columns)
	present = [c for c in DROP_COLS if c in cols]
	missing = [c for c in DROP_COLS if c not in cols]

	if missing:
		print("Uyarı: Aşağıdaki sütunlar bulunamadı ve kaldırılamadı:")
		for c in missing:
			print(f" - {c}")

	if dst.exists():
		dst.unlink()

	chunksize = 100_000
	first = True
	for chunk in pd.read_csv(src, chunksize=chunksize, low_memory=False):
		chunk = chunk.drop(columns=DROP_COLS, errors="ignore")
		chunk.to_csv(dst, mode="w" if first else "a", index=False, header=first)
		first = False

	print(f"Tamamlandı. Çıktı: {dst}")
	if present:
		print("Kaldırılan sütunlar:")
		for c in present:
			print(f" - {c}")


if __name__ == "__main__":
	main()
