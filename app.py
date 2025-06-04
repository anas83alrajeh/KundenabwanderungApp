import streamlit as st
import pandas as pd
import joblib
import numpy as np
import io

# === Modell laden ===
modell_pfad = "models/xgboost_model.joblib"
modell = joblib.load(modell_pfad)

# === Merkmale ===
modell_merkmale = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges',
    'TotalCharges', 'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
    'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes'
]

# === Titel ===
st.markdown(
    "<h3 style='text-align: center; color: white;'>Telco Kundenabwanderung Vorhersage</h3>",
    unsafe_allow_html=True
)

# === Auswahl: Einzel oder Datei oder Reinigung ===
st.markdown("<p style='font-size:14px; color:gray;'>Erstellt von Anas Al Rajeh</p>", unsafe_allow_html=True)
st.markdown("<p style='font-size:14px; color:gray;'>Kontakt: <a href='mailto:anasalrajeh9@gmail.com'>anasalrajeh9@gmail.com</a></p>", unsafe_allow_html=True)

wahl = st.radio("Eingabemodus auswählen", ["Einzelne Eingabe", "Datei zum Verarbeiten hochladen", "Daten bereinigen"])


# === Funktion für Einzeldateneingabe ===
def nutzereingabe():
    gender = st.selectbox("Geschlecht", [0, 1], format_func=lambda x: "Männlich" if x == 1 else "Weiblich")
    st.caption("Geschlecht des Kunden: 0 = Weiblich, 1 = Männlich")

    SeniorCitizen = st.selectbox("Senior (65+)?", [0, 1])
    st.caption("Ist der Kunde 65 Jahre oder älter? 1 = Ja, 0 = Nein")

    Partner = st.selectbox("Hat Partner?", [0, 1])
    st.caption("Hat der Kunde einen Ehepartner oder Lebenspartner? 1 = Ja, 0 = Nein")

    Dependents = st.selectbox("Hat Angehörige?", [0, 1])
    st.caption("Hat der Kunde Angehörige wie Kinder oder andere Abhängige?")

    tenure = st.slider("Vertragsdauer (Monate)", 0, 72, 12)
    st.caption("Wie viele Monate war der Kunde bereits beim Anbieter?")

    PhoneService = st.selectbox("Telefonservice", [0, 1])
    st.caption("Hat der Kunde einen aktiven Telefonservice?")

    OnlineSecurity = st.selectbox("Online-Sicherheit", [0, 1])
    st.caption("Verfügt der Kunde über einen Online-Sicherheitsdienst wie Firewall oder Antivirus?")

    OnlineBackup = st.selectbox("Online-Backup", [0, 1])
    st.caption("Nutzt der Kunde Online-Datensicherung?")

    DeviceProtection = st.selectbox("Geräteschutz", [0, 1])
    st.caption("Hat der Kunde eine Geräteschutzversicherung?")

    TechSupport = st.selectbox("Technischer Support", [0, 1])
    st.caption("Nutzt der Kunde technischen Support-Dienstleistungen?")

    StreamingTV = st.selectbox("Streaming-TV", [0, 1])
    st.caption("Verwendet der Kunde Streaming-TV-Dienste?")

    StreamingMovies = st.selectbox("Streaming-Filme", [0, 1])
    st.caption("Verwendet der Kunde Streaming-Filmangebote?")

    PaperlessBilling = st.selectbox("Papierlose Rechnung", [0, 1])
    st.caption("Erhält der Kunde Rechnungen nur digital ohne Papier?")

    MonthlyCharges = st.number_input("Monatliche Kosten (€)", min_value=0.0, value=50.0)
    st.caption("Wie viel zahlt der Kunde monatlich?")

    TotalCharges = st.number_input("Gesamtkosten (€)", min_value=0.0, value=500.0)
    st.caption("Wie viel hat der Kunde insgesamt bezahlt?")

    internet_optionen = {
        'DSL': [1, 0, 0],
        'Fiber optic': [0, 1, 0],
        'Keine': [0, 0, 1]
    }
    internet = st.selectbox("Internetservice", list(internet_optionen.keys()))
    st.caption("Welchen Internetservice verwendet der Kunde?")
    InternetService_DSL, InternetService_Fiber_optic, InternetService_No = internet_optionen[internet]

    vertrag_optionen = {
        'Monatlich': [1, 0, 0],
        'Ein Jahr': [0, 1, 0],
        'Zwei Jahre': [0, 0, 1]
    }
    vertrag = st.selectbox("Vertragsart", list(vertrag_optionen.keys()))
    st.caption("Wie lange läuft der Vertrag des Kunden?")
    Contract_Month_to_month, Contract_One_year, Contract_Two_year = vertrag_optionen[vertrag]

    zahlung_optionen = {
        'Banküberweisung (automatisch)': [1, 0, 0, 0],
        'Kreditkarte (automatisch)': [0, 1, 0, 0],
        'Elektronischer Scheck': [0, 0, 1, 0],
        'Post-Scheck': [0, 0, 0, 1]
    }
    zahlung = st.selectbox("Zahlungsmethode", list(zahlung_optionen.keys()))
    st.caption("Welche Zahlungsmethode verwendet der Kunde?")
    PaymentMethod_Bank_transfer, PaymentMethod_Credit_card, PaymentMethod_Electronic_check, PaymentMethod_Mailed_check = zahlung_optionen[zahlung]

    lines_optionen = {
        'Nein': [1, 0, 0],
        'Kein Telefondienst': [0, 1, 0],
        'Ja': [0, 0, 1]
    }
    lines = st.selectbox("Mehrere Leitungen", list(lines_optionen.keys()))
    st.caption("Verfügt der Kunde über mehrere Telefonleitungen?")
    MultipleLines_No, MultipleLines_No_phone, MultipleLines_Yes = lines_optionen[lines]

    daten = np.array([[gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
                       OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
                       StreamingTV, StreamingMovies, PaperlessBilling, MonthlyCharges,
                       TotalCharges, InternetService_DSL, InternetService_Fiber_optic, InternetService_No,
                       Contract_Month_to_month, Contract_One_year, Contract_Two_year,
                       PaymentMethod_Bank_transfer, PaymentMethod_Credit_card,
                       PaymentMethod_Electronic_check, PaymentMethod_Mailed_check,
                       MultipleLines_No, MultipleLines_No_phone, MultipleLines_Yes]])

    return pd.DataFrame(daten, columns=modell_merkmale)

# === Einzel-Modus ===
if wahl == "Einzelne Eingabe":
    eingabe_df = nutzereingabe()
    if st.button("Abwanderung vorhersagen"):
        vorhersage = modell.predict(eingabe_df)[0]
        if vorhersage == 1:
            st.error("Es wird erwartet, dass der Kunde abwandert.")
        else:
            st.success("Es wird erwartet, dass der Kunde bleibt.")

# === Datei-Modus (CSV oder Excel) ===
elif wahl == "Datei zum Verarbeiten hochladen":
    datei = st.file_uploader("CSV- oder Excel-Datei mit Kundendaten hochladen", type=["csv", "xlsx"])

    if datei is not None:
        try:
            if datei.name.endswith('.csv'):
                df = pd.read_csv(datei)
            else:
                df = pd.read_excel(datei)
        except Exception as e:
            st.error(f"Fehler beim Einlesen der Datei: {e}")
            df = None

        if df is not None:
            if set(modell_merkmale).issubset(df.columns):
                prognosen = modell.predict(df[modell_merkmale])
                df["Prognose"] = prognosen

                st.success("Vorhersagen erfolgreich berechnet.")
                st.dataframe(df.head())

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Vorhersagen', index=False)
                output.seek(0)

                st.download_button(
                    label="Ergebnisse als Excel-Datei herunterladen",
                    data=output,
                    file_name="vorhersagen.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                fehlende = set(modell_merkmale) - set(df.columns)
                st.error(f"Die Datei fehlt folgende Spalten:\n{fehlende}")

# === Reinigungsmodus ===
elif wahl == "Daten bereinigen":
    hochgeladene_datei = st.file_uploader("CSV-Datei mit unbearbeiteten Kundendaten hochladen", type=["csv"])

    if hochgeladene_datei is not None:
        df_roh = pd.read_csv(hochgeladene_datei)

        # Datenbereinigung
        df_roh['TotalCharges'] = pd.to_numeric(df_roh['TotalCharges'], errors='coerce')
        df_roh.dropna(inplace=True)

        binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            df_roh[col] = df_roh[col].map({'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1})

        df_roh['SeniorCitizen'] = df_roh['SeniorCitizen'].astype(int)

        kategorische_spalten = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                'Contract', 'PaymentMethod']

        df_verarbeitet = pd.get_dummies(df_roh, columns=kategorische_spalten)

        fehlende_spalten = set(modell_merkmale) - set(df_verarbeitet.columns)
        for col in fehlende_spalten:
            df_verarbeitet[col] = 0

        df_verarbeitet = df_verarbeitet[modell_merkmale]

        st.success("Daten erfolgreich bereinigt!")
        st.dataframe(df_verarbeitet.head())

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_verarbeitet.to_excel(writer, sheet_name='BereinigteDaten', index=False)
        output.seek(0)

        st.download_button(
            label="Bereinigte Daten herunterladen",
            data=output,
            file_name="bereinigte_daten.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
