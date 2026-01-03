import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import (f1_score, precision_score, recall_score, accuracy_score, roc_curve, roc_auc_score, 
average_precision_score, precision_recall_curve)
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle


# Glob ile MachineLearningCVE klasöründeki CSV dosyalarını al
csv_files = glob.glob('C:\\Users\\kul38\\Desktop\\IDS Project\\training\\test3\\MachineLearningCVE\\*.csv')

print(f"Bulunan dosyalar: {len(csv_files)}")
if not csv_files:
    print("HATA: CSV dosyası bulunamadı!")
    print(f"Çalışma dizini: {os.getcwd()}")
    exit()

# Tüm CSV dosyalarını oku
dfs = [pd.read_csv(f) for f in csv_files]

# Pandas concat ile birleştir
merged_df = pd.concat(dfs, ignore_index=True)

# CSV olarak kaydet
merged_df.to_csv('merged_data.csv', index=False)

print(f"Birleştirme tamamlandı: {merged_df.shape}")

