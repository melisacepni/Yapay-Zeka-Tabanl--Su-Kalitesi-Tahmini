#Streamlit ile Su Kalitesi Analizi ve Tahmini Uygulaması

#Uygulama Hakkında
Bu Streamlit uygulaması, su kalitesi verilerini analiz etmek ve suyun içilebilir olup olmadığını tahmin etmek için bir makine öğrenimi modeli kullanır. Random Forest algoritması ile geliştirilen bu uygulama, kullanıcı dostu bir arayüz ve modern bir tasarımla bilimsel verileri kolayca yorumlamanızı sağlar.

#Uygulama Özellikleri

Kullanıcı Dostu Arayüz:
Kullanıcılar, giriş parametrelerini kolayca sağlayarak tahmin alabilirler.
Makine Öğrenimi Entegrasyonu:
Random Forest algoritması ile suyun içilebilirliği tahmin edilir.
Veri Temizleme:
Eksik değerler sütun medyanları ile doldurularak model performansı optimize edilir.
Şık Tasarım:
Gradient arka plan, özelleştirilmiş yazı tipleri ve buton stili ile estetik bir görünüm sunar.

#Kurulum ve Çalıştırma
Gerekli Kütüphaneler
Uygulamayı çalıştırmadan önce aşağıdaki Python kütüphanelerini yükleyin:
pip install streamlit pandas numpy scikit-learn

Başlatma
Uygulamayı başlatmak için aşağıdaki adımları izleyin:

app.py dosyasını bir metin editörü veya IDE ile açın.
Veri yolunu (data_path) kendi dosya yapınıza uygun şekilde değiştirin. Varsayılan yol:
C:\Users\USER\Desktop\Miuul Proje\water_potability.csv


#Uygulamayı başlatın:
streamlit run app.py
Adım Adım Açıklama
1. Uygulama Başlığı ve Stil
Uygulama, gradient bir arka plan ve modern CSS kullanılarak özelleştirilmiştir.
Başlık ve Açıklama: Kullanıcıya uygulamanın işlevselliği ve amacı hakkında bilgi verir.
2. Veri Yükleme ve Temizleme
Kullanıcı, su kalitesi verilerinin bulunduğu bir CSV dosyasını yükler.
Eksik Veriler: Medyan değerler ile doldurulur.
Bu adım, verilerin analiz ve model eğitimi için uygun hale getirilmesini sağlar.
3. Model Eğitimi
Veri seti, özellikler (X) ve hedef (y) olarak ayrılır.
Eğitim ve Test Bölünmesi: Veriler %70 eğitim, %30 test seti olarak ayrılır.
Random Forest Classifier:
Model, eğitim seti üzerinde eğitilir.
Test seti üzerinde doğruluk skoru hesaplanır.
4. Tahmin Arayüzü
Kullanıcı, su kalitesi parametrelerini girer (örneğin: pH, sertlik, kloramin, vb.).
Model, girilen verilere göre suyun içilebilirlik durumunu tahmin eder:
0 = İçilemez
1 = İçilebilir
Tahmin sonuçları, olasılık yüzdesi ile birlikte gösterilir.
5. Dinamik Özellikler
Tahmin Butonu: Kullanıcı "Tahmin Yap" butonuna tıklayarak sonuç alır.
Önbellekleme: Veri yükleme ve model eğitimi işlemleri optimize edilir.
Kullanıcı Deneyimi
Anlaşılır Sonuçlar:
Tahmin sonuçları renk kodlarıyla (ör. yeşil = içilebilir) görselleştirilir.

#İnteraktif Arayüz:
Kullanıcılar parametreleri kolayca girip tahmin yapabilirler.
Örnek Kod Parçaları
Veri Yükleme
import pandas as pd

data_path = "C:\\Users\\USER\\Desktop\\Miuul Proje\\water_potability.csv"
df = pd.read_csv(data_path)
df.fillna(df.median(), inplace=True)
Model Eğitimi
python
Kodu kopyala
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
Streamlit Arayüzü
python
Kodu kopyala
import streamlit as st

st.title("Su Kalitesi Analizi ve Tahmini")
st.markdown("Su kalitesi parametrelerini girerek içilebilirlik tahmini yapabilirsiniz.")

# Kullanıcı girişleri
pH = st.slider("pH Değeri", 0.0, 14.0, 7.0)
hardness = st.number_input("Sertlik (mg/L)")
chloramines = st.number_input("Kloramin (mg/L)")

if st.button("Tahmin Yap"):
    prediction = model.predict([[pH, hardness, chloramines]])[0]
    if prediction == 1:
        st.success("Tahmin: İçilebilir 🟢")
    else:
        st.error("Tahmin: İçilemez 🔴")
Notlar
Veri yolu (data_path) doğru bir şekilde ayarlanmalıdır.
Veri setinde eksik veri varsa, bu eksiklikler otomatik olarak doldurulacaktır.






