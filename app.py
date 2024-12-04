import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sayfa stilini özelleştirme (arka planı gradient, yazıları lacivert yapmak için)
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(135deg, #87CEEB, #1E3A8A); /* Mavi gradient arka plan */
            color: #FFFFFF; /* Beyaz yazı rengi */
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #1E3A8A;
            color: white;
            font-size: 16px;
            padding: 12px 30px;
            border-radius: 8px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #1D4ED8;
        }
        .stTitle, .stSubheader, .stText {
            color: #FFFFFF;
            font-family: 'Arial', sans-serif;
        }
        .stNumberInput input {
            font-size: 16px;
        }
        .stTextInput input {
            font-size: 16px;
        }
        .stTextArea input {
            font-size: 16px;
        }
        .stSubheader {
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True
)

# Başlık
st.title("Su Kalitesi Analizi ve Tahmini")
st.write("Bu uygulama, su kalitesi verilerini analiz etmenize ve içilebilirlik tahminleri yapmanıza yardımcı olur.")

# Veri yükleme ve temizleme
@st.cache_data
def load_and_process_data():
    # Veriyi yükle
    df = pd.read_csv(r"C:\Users\USER\Desktop\Miuul Proje\water_potability.csv", index_col=0)
    
    df = df.reset_index()

    # Eksik değer doldurma
    df["ph"] = df["ph"].fillna(value=df["ph"].median())
    df["Sulfate"] = df["Sulfate"].fillna(value=df["Sulfate"].median())
    df["Trihalomethanes"] = df["Trihalomethanes"].fillna(value=df["Trihalomethanes"].median())
    
    return df

data = load_and_process_data()

# Model eğitimi
@st.cache_resource
def train_model(data):
    X = data.drop('Potability', axis=1).values
    y = data['Potability'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    
    # Random Forest modeli
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Test doğruluğu
    accuracy = accuracy_score(y_test, model.predict(X_test))
    
    return model, accuracy

model, accuracy = train_model(data)

# Veri gösterimi
# st.write("Veri Kümesi:")
# st.write(data.head())

# st.write(f"Model Test Doğruluğu: %{accuracy * 100:.2f}")

# Tahmin arayüzü
st.subheader("İçilebilirlik Tahmini")
st.write("Aşağıdaki değerleri girerek suyun içilebilirlik durumunu tahmin edin:")

# Kullanıcıdan giriş alma
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
hardness = st.number_input("Hardness", min_value=0.0, value=150.0, step=0.1)
solids = st.number_input("Solids", min_value=0.0, value=20000.0, step=1.0)
chloramines = st.number_input("Chloramines", min_value=0.0, value=6.0, step=0.1)
sulfate = st.number_input("Sulfate", min_value=0.0, value=300.0, step=0.1)
conductivity = st.number_input("Conductivity", min_value=0.0, value=400.0, step=1.0)
organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=10.0, step=0.1)
trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=80.0, step=0.1)
turbidity = st.number_input("Turbidity", min_value=0.0, value=4.0, step=0.1)

# Tahmin butonu
if st.button("Tahmin Yap"):
    # Kullanıcı girişlerini modele uygun forma dönüştür
    input_data = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
    
    # Tahmini hesapla
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    # Sonucu göster
    if prediction == 1:
        st.success(f"Bu su içilebilir! (%{probability * 100:.2f} olasılıkla)")
    else:
        st.error(f"Bu su içilemez! (%{(1 - probability) * 100:.2f} olasılıkla)")
