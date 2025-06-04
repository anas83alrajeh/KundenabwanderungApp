# KundenabwanderungApp
Vorhersage der Kundenabwanderung im Telekommunikationssektor: "Vergleich von Decision Tree und XGBoost zur Auswahl des besten Machine-Learning-Modells"
 ==============================================================================
 # Telco Kundenabwanderungsvorhersage (Churn Prediction)
 ==============================================================================

Dieses Projekt zielt darauf ab, vorherzusagen, ob ein Kunde bei einem Telekommunikationsunternehmen kündigt (Churn), basierend auf Kundendaten. Die Lösung enthält:

    - Explorative Datenanalyse (EDA)
    - Datenvorverarbeitung & Kodierung
    - Behandlung von unausgeglichenen Klassen mit SMOTE
    - Modelltraining und Evaluierung (Decision Tree & XGBoost)
    - Modellinterpretierbarkeit mit SHAP
    - Interaktive Web-App mit Streamlit


 ==============================================================================
 # Projektordner/
 ==============================================================================
│
├── data/
│ ├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Rohdaten
│ └── processed_telco_churn_balanced.csv # Aufbereitete und balancierte Daten
│
├── models/
│ └── xgboost_model.joblib oder decision_tree_model.joblib (automatisch gespeichert)
│
├── app.py # Streamlit-App zur Vorhersage
├── preprocessed_telco_churn_app.py # Skript zum Training und Modellauswahl
├── requirements.txt # Benötigte Python-Bibliotheken
└── README.md # Projektdokumentation


 ==============================================================================
 # Hauptskripte
 ==============================================================================
1. Datenvorverarbeitung und Balancierung

python churn_data_preprocessing.py
    - Wandelt kategorische Merkmale um
    - Bereinigt fehlende Werte
    - Balanciert das Datenset mit SMOTE
    - Speichert die verarbeiteten Daten unter data/processed_telco_churn_balanced.csv

2. Modelltraining & Auswahl

python churn_training_pipeline.py
    - Trainiert zwei Modelle: DecisionTreeClassifier und XGBoost
    - Bewertet mit Metriken wie ROC AUC
    - Speichert das beste Modell im Ordner models/
    - Visualisiert ROC-Kurve & Confusion Matrix
    - Führt SHAP-Analyse durch (falls XGBoost gewählt wurde)

3. Web-App starten

streamlit run telco_churn_app.py
    - Benutzerfreundliche Oberfläche zur Eingabe von Kundendaten
    - Gibt Vorhersage zurück: Kunde wird abwandern / bleibt
    - Zeigt (optional) interpretierbare Erklärungen mit SHAP an

 ==============================================================================
 Genutzte Merkmale
 ==============================================================================

    - Das Modell verwendet unter anderem folgende Merkmale:
    - Demografisch: Geschlecht, Partner, Kinder, Seniorenstatus
    - Vertrag: Vertragslaufzeit, Zahlungsmethode, Internetdienst
    - Nutzung: OnlineSecurity, TechSupport, Streaming, Telefonservice
    - Numerisch: tenure, MonthlyCharges, TotalCharges


 ==============================================================================
  Evaluierung
 ==============================================================================

Die Modelle werden bewertet mit:
    - ROC AUC Score
    - Confusion Matrix
    - Classification Report (Präzision, Recall, F1-Score)
    - SHAP-Werte zur Modellinterpretation


 ==============================================================================
 Anforderungen
 ==============================================================================

    - Alle Bibliotheken befinden sich in requirements.txt. Die wichtigsten sind:
    - pandas, numpy, scikit-learn
    - xgboost, shap, imblearn
    - streamlit, matplotlib, seaborn


 ==============================================================================
 ==============================================================================
  Projekt ausführen
 ==============================================================================
 ==============================================================================

        Anaconda Prompt
cd /d D:\EducX\ML\Woche4\Telco-Churn-Prediction
streamlit run app.py   

 
 ==============================================================================
 Lizenz
 ==============================================================================
Dieses Projekt steht unter der MIT-Lizenz. Siehe LICENSE für weitere Details.

==============================================================================
 Kontakt
 ==============================================================================

Autor: Anas Al Rajeh.
E-Mail: anasalrajeh9@gmail.com
