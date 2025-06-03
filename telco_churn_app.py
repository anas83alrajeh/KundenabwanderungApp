# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 14:25:15 2025
@author: anasa
"""

import os
import joblib
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
import shap

# =============================================================================
# === Funktion zum Löschen alter Modell-Dateien ===
# =============================================================================
def delete_old_models(model_folder='models'):
    # Prüfen, ob der Ordner existiert
    if os.path.exists(model_folder):
        # Alle Dateien im Ordner durchlaufen
        for f in os.listdir(model_folder):
            # Wenn die Datei mit '.joblib' endet (Modell-Dateien)
            if f.endswith('.joblib'):
                # Datei löschen
                os.remove(os.path.join(model_folder, f))
    else:
        # Falls Ordner nicht existiert, wird er erstellt
        os.makedirs(model_folder)

# =============================================================================
# === Ordner zur Speicherung der Modelle ===
# =============================================================================
model_folder = 'models'
# Alte Modelle löschen, bevor neues Training beginnt
delete_old_models(model_folder)

# =============================================================================
# === Daten laden ===
# =============================================================================
data = pd.read_csv('data/processed_telco_churn_balanced.csv')

# Form der Daten ausgeben (Zeilen, Spalten)
print(f"Form der Daten: {data.shape}")  
# Ersten 5 Zeilen der Daten anzeigen
print(data.head())
# Info zu Datentypen und fehlenden Werten ausgeben
print(data.info())

# Anzahl der Instanzen je Klasse im Zielattribut 'Churn'
print(data['Churn'].value_counts())
# Anteil der Klassen in Prozent
print(data['Churn'].value_counts(normalize=True) * 100)

# Visualisierung: Verteilung der Zielvariable nach Balancierung
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=data)
plt.title('Verteilung der Zielvariable Churn nach Balancierung')
plt.xlabel('Churn')
plt.ylabel('Anzahl')
plt.show()

# =============================================================================
# Statistische Übersicht der numerischen Variablen
# =============================================================================
print(data.describe())

# Liste der numerischen Spalten zur Visualisierung
numerische_spalten = ['tenure', 'MonthlyCharges', 'TotalCharges']
for spalte in numerische_spalten:
    plt.figure(figsize=(8,4))
    # Histogramm mit Dichtekurve (KDE) für jede numerische Variable
    sns.histplot(data[spalte], bins=30, kde=True)
    plt.title(f'Verteilung der Variable: {spalte}')
    plt.xlabel(spalte)
    plt.ylabel('Häufigkeit')
    plt.show()

# Klassenverteilung zählen
klassen_verteilung = Counter(data['Churn'])
gesamt = sum(klassen_verteilung.values())
print("\nKlassenverteilung in Prozent:")
# Prozentuale Verteilung je Klasse ausgeben
for klasse, anzahl in klassen_verteilung.items():
    print(f"Klasse {klasse}: {anzahl} ({(anzahl/gesamt)*100:.2f}%)")

from sklearn.model_selection import train_test_split
# Merkmale (Features) von Zielvariable trennen
X = data.drop('Churn', axis=1)
y = data['Churn']

# Trainings- und Testdaten aufteilen mit stratified sampling für gleiche Klassenverteilung
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Klassenverteilung im Trainings- und Testset ausgeben
print("Verteilung der Klassen im Trainingsset:", Counter(y_train))
print("Verteilung der Klassen im Testset:", Counter(y_test))

# =============================================================================
# === Decision Tree Modell trainieren ===
# =============================================================================
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Vorhersagen für Testdaten
y_pred = dt_model.predict(X_test)
# Wahrscheinlichkeiten für positive Klasse
y_pred_proba = dt_model.predict_proba(X_test)[:, 1]

# Klassifikationsbericht ausgeben (Precision, Recall, F1-Score)
print("Classification Report für Decision Tree:\n", classification_report(y_test, y_pred))
# Konfusionsmatrix anzeigen
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# ROC AUC Score berechnen für Decision Tree
roc_auc_dt = roc_auc_score(y_test, y_pred_proba)
print("ROC AUC Score (Decision Tree):", roc_auc_dt)

# =============================================================================
# === XGBoost Modell trainieren ===
# =============================================================================
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Vorhersagen und Wahrscheinlichkeiten für XGBoost
y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:,1]

print("\n Klassifikationsbericht für XGBoost:")
print(classification_report(y_test, y_pred_xgb))
# ROC AUC Score für XGBoost berechnen
roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
print(f"AUC-ROC Score (XGBoost): {roc_auc_xgb:.4f}")

# =============================================================================
# === Auswahl und Speichern des besten Modells basierend auf AUC ROC ===
# =============================================================================
best_model = None
best_model_name = ""
best_auc = 0.0

if roc_auc_xgb > roc_auc_dt:
    best_model = xgb_model
    best_model_name = "xgboost_model.joblib"
    best_auc = roc_auc_xgb
else:
    best_model = dt_model
    best_model_name = "decision_tree_model.joblib"
    best_auc = roc_auc_dt

# Pfad zum Speichern des besten Modells
save_path = os.path.join(model_folder, best_model_name)
joblib.dump(best_model, save_path)
print(f"\nBestes Modell '{best_model_name}' wurde gespeichert mit AUC: {best_auc:.4f}")

# =============================================================================
# === Beispiel: Laden des Modells ===
# =============================================================================
# loaded_model = joblib.load(save_path)

# =============================================================================
# === Optional: Visualisierungen ===
# =============================================================================

# ROC-Kurvenvergleich Decision Tree vs. XGBoost
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)

plt.figure(figsize=(8,6))
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_dt:.2f})',color='darkorange')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})', color='darkgreen')
plt.plot([0,1], [0,1], color='gray', linestyle='--')  # Zufallslinie
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Vergleich: Decision Tree vs XGBoost')
plt.legend()
plt.grid()
plt.show()

# Konfusionsmatrix des besten Modells anzeigen
best_pred = best_model.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, best_pred, cmap='Blues')
plt.title(f'Confusion Matrix für bestes Modell ({best_model_name})')
plt.grid(False)
plt.show()

# =============================================================================
# === SHAP-Erklärungen nur, wenn XGBoost bestes Modell ist ===
# =============================================================================
if best_model_name == "xgboost_model.joblib":
    # SHAP TreeExplainer für XGBoost initialisieren
    explainer = shap.TreeExplainer(best_model)
    # SHAP-Werte für Testdaten berechnen
    shap_values = explainer.shap_values(X_test)
    # Zusammenfassungsplot als Balkendiagramm (Feature-Wichtigkeit)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    # Zusammenfassungsplot als Punktewolke (Detailansicht)
    shap.summary_plot(shap_values, X_test)
    # JavaScript für interaktive Plots initialisieren (im Notebook)
    shap.initjs()
    # Detaillierte Force-Plot-Visualisierung für eine einzelne Beobachtung
    shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0], matplotlib=True)
