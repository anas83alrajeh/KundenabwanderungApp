# -*- coding: utf-8 -*-
"""
Created on Fri May 30 16:01:34 2025

@author: anasa
"""
# =============================================================================
# Bibliotheken importieren / Import libraries
# =============================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE

# =============================================================================
# 0. Datei löschen falls vorhanden / Delete existing processed file if exists
# =============================================================================
file_path = 'data/processed_telco_churn_balanced.csv'
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"Alte Datei wurde gelöscht: {file_path}")

# =============================================================================
# CSV-Datei laden / Load CSV file
# =============================================================================
data = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# =============================================================================
# 1. Überblick über die Daten (EDA) / Exploratory Data Analysis
# =============================================================================
print("Form der Daten:", data.shape)
print("Erste 5 Zeilen:")
print(data.head())

print("\nDatentypen und fehlende Werte:")
print(data.info())

print("\nVerteilung der Zielvariable 'Churn':")
print(data['Churn'].value_counts())
print("\nProzentuale Verteilung:")
print(data['Churn'].value_counts(normalize=True) * 100)

sns.countplot(x='Churn', data=data)
plt.title('Verteilung der Zielvariable Churn (vor Balancierung)')
plt.show()

print("\nDeskriptive Statistik der numerischen Variablen:")
print(data.describe())

# Boxplots zur Erkennung von Ausreißern
numerische_spalten = ['tenure', 'MonthlyCharges', 'TotalCharges']
for spalte in numerische_spalten:
    plt.figure(figsize=(8,4))
    sns.boxplot(x=data[spalte])
    plt.title(f'Boxplot für {spalte}')
    plt.show()

# =============================================================================
# 2. Datenvorbereitung: Fehlende Werte und Typumwandlung / Data preparation
# =============================================================================
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

# =============================================================================
# 3. Kategoriale Variablen kodieren / Encoding categorical variables
# =============================================================================
# Spalten mit 'Yes', 'No', 'No internet service' in 0/1 umwandeln
binary_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
               'StreamingTV', 'StreamingMovies']

for col in binary_cols:
    data[col] = data[col].replace({'Yes':1, 'No':0, 'No internet service':0})

# Label Encoding für binäre Spalten
binäre_spalten = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                  'PaperlessBilling', 'Churn']

le = LabelEncoder()
for spalte in binäre_spalten:
    data[spalte] = le.fit_transform(data[spalte])

# One-Hot Encoding für Mehrkategorien-Spalten
multi_kategoriale_spalten = ['InternetService', 'Contract', 'PaymentMethod', 'MultipleLines']
data = pd.get_dummies(data, columns=multi_kategoriale_spalten)

# =============================================================================
# 4. Nicht relevante Spalte entfernen / Drop irrelevant column
# =============================================================================
data.drop('customerID', axis=1, inplace=True)

# =============================================================================
# 5. Verteilung der Klassen vor Balancierung anzeigen / Show class distribution before balancing
# =============================================================================
X = data.drop('Churn', axis=1)
y = data['Churn']

print("\nVerteilung der Klassen vor der Balancierung:")
print(Counter(y))

# =============================================================================
# Optional: Visualisierung der Klassenverteilung vor Balancierung
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title('Klassenverteilung vor Balancierung')
plt.show()

# =============================================================================
# 6. Balancierung der Daten mit SMOTE / Balance data with SMOTE
# =============================================================================
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nVerteilung der Klassen nach der Balancierung:")
print(Counter(y_resampled))

# =============================================================================
# Optional: Visualisierung der Klassenverteilung nach Balancierung
plt.figure(figsize=(6,4))
sns.countplot(x=y_resampled)
plt.title('Klassenverteilung nach Balancierung (SMOTE)')
plt.show()

# =============================================================================
# 7. Resampled Daten in DataFrame zusammenführen und speichern / Combine and save resampled data
# =============================================================================
df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                         pd.DataFrame(y_resampled, columns=['Churn'])], axis=1)

df_resampled.to_csv(file_path, index=False)
print(f"\nAusgewogene Daten wurden gespeichert in '{file_path}'")

# =============================================================================
# 8. Klassenverhältnis nach Balancierung (in %) / Class ratio after balancing (in %)
# =============================================================================
klassen_verhältnis = Counter(y_resampled)
gesamt = sum(klassen_verhältnis.values())
print("\nKlassenverhältnis nach Balancierung (in %):")
for klasse, anzahl in klassen_verhältnis.items():
    print(f"Klasse {klasse}: {anzahl} ({(anzahl/gesamt)*100:.2f}%)")
