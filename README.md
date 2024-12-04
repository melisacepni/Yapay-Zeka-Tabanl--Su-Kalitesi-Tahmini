# YAPAY ZEKA TABANLI SU KALİTESİ TAHMİNİ

Miuul Data Scientist Bootcamp katılımcısı olarak, 5 kişilik bir grup olarak yaptığımız bitirme projesinin konusu "Yapay Zeka Tabanlı Su Kalitesi Tahmini"dir.

-------- 

Günümüzde su kirliliği, insan sağlığı ve ekosistemler üzerinde ciddi tehditler oluşturmaktadır. Özellikle artan sanayileşme ve hızlı nüfus artışı nedeniyle su kaynaklarının kalitesi hızla düşmekte, bu durum hem doğal yaşamı hem de insan sağlığını olumsuz etkilemektedir (Khatri & Tyagi, 2015). Su kaynaklarının izlenmesi ve korunması, sürdürülebilir kalkınma hedefleri doğrultusunda kritik bir adım olarak değerlendirilmektedir. Ancak, geleneksel su kalitesi izleme yöntemleri çoğunlukla pahalı, zaman alıcı ve sürekli izleme gereksinimlerini karşılayamamaktadır (Zhu et al., 2022). Bu noktada, yapay zeka ve makine öğrenmesi teknikleri su kalitesinin izlenmesinde yeni ve daha etkin çözümler sunmaktadır. Özellikle su kalitesi parametrelerini tahmin etmek için kullanılan makine öğrenmesi modelleri, büyük veri setlerinden elde edilen bilgilerle eğitilerek gelecekteki kirlilik durumlarını öngörebilmektedir. Bu bağlamda, su kalitesini belirlemek için kullanılan yapay zeka tabanlı algoritmalar, hem daha hızlı hem de düşük maliyetli izleme çözümleri sunarak geleneksel yöntemlere alternatif oluşturmaktadır (Yan et al. 2024).

--------

Projemizin genel amacı, makine öğrenmesi algoritmalarını kullanarak su kalitesi parametrelerine dayalı tahmin modelleri geliştirerek, su kaynaklarının daha etkin bir şekilde izlenmesini sağlamaktır. Bu çalışma, genel su izleme süreçlerini optimize edecek ve farklı bölgelerde de uygulanabilir, genellenebilir bir çözüm sunmayı hedeflemektedir. Proje sonucunda geliştirilen modelin su kalitesi izleme sistemlerine entegre edilmesi durumunda, su kalitelerinin korunması ve yönetimi konusunda model, yenilikçi bir çözüm sunabilecektir.

--------

## 3 aşamadan oluşan projemizin aşamaları şu şekildedir;

#### - Aşama 1:
Veri Seti Analizi ve İnceleme
Veri Temizleme
Veri Görselleştirme ve Keşifsel Veri Analizi

#### - Aşama 2:

Model Seçimi ve Uygulaması
Model Performans Değerlendirmeleri
Model İyileştirme ve Karşılaştırma

#### - Aşama 3:

Nihai Modelin Seçimi ve Son Testler

-------- 

## Veri Seti Hikayesi:

Veri bilimi projeleri gerçekleştirilirken sıklıkla kullanılan veri havuzu platformu Kaggle.com sitesinden elde edilen ‘water_potability.csv’ dosyasında, 3276 farklı su kaynağına ait kalite ölçütleri bulunmaktadır. Toplamda 9 adet bağımsız değişken doğrultusunda, suyun içilebilme durumu belirtilmektedir.

#### Değişkenler:
1-	 pH: Suyun asidik veya bazik durumunu gösterir. WHO’ya göre ideal değer aralığı 6.5-8.5’tir.
2-	Hardness: Suyun içerdiği kalsiyum ve magnezyum tuzlarından kaynaklanır.
3-	Solids (Total dissolved solids - TDS): Suda çözünmüş mineral ve tuzların toplamıdır. İdeal limit 500 mg/L, maksimum limit 1000 mg/L'dir.
4-	Chloramines: Suyun dezenfekte edilmesi için kullanılan klor ve amonyak bileşimidir. 4 mg/L'ye kadar güvenlidir.
5-	Sulfate: Doğal olarak toprak ve kayalarda bulunur. Genellikle 3-30 mg/L aralığındadır, bazı bölgelerde 1000 mg/L’ye kadar çıkabilir.
6-	Conductivity: Çözünmüş iyonların miktarını ölçer. WHO standardına göre 400 μS/cm’yi aşmamalıdır.
7-	Organic Carbon: Çözünmüş organik maddelerin karbon miktarını ifade eder. İçme suyu için sınır 2 mg/L’dir.
8-	Trihalomethanes (THM): Klor ile işlem gören suda oluşabilir. 80 ppm’ye kadar güvenlidir.
9-	Turbidity: Suda askıda bulunan katı maddelerden kaynaklanır. WHO’ya göre sınır değeri 5 NTU’dur.
10-	Potability (Potability): Suyun içilebilir olup olmadığını gösterir. 1 içilebilir, 0 içilemez anlamına gelir. Bu veriler, suyun kalitesini ve içme suyuna uygunluğunu değerlendirmek için kullanılabilir.

--------

## Genel Bakış:

Gözlem sayısı: 3276
Değişken sayısı: 10
Kategorik değişken sayısı: 1
Sayısal değişken sayısı: 9
Sayısal görünen ama kategorik değişken sayısı: 1
Kategorik görünen ama sayısal değişken sayısı: 0 

--------

Veri setinde toplam 1434 eksik değer bulunmaktadır ve bu eksik değerlerin doldurulması noktasında ortalama ile doldurma yöntemi kullanılmıştır. Bunun yanında veri setimizde aykırı değer bulunmadığı için, bu konuda herhangi bir işlem uygulanmamıştır.

--------

## Streamlit ile Su Kalitesi Analizi ve Tahmini Uygulaması

#### Uygulama Hakkında
Bu Streamlit uygulaması, su kalitesi verilerini analiz etmek ve suyun içilebilir olup olmadığını tahmin etmek için bir makine öğrenimi modeli kullanır. Random Forest algoritması ile geliştirilen bu uygulama, kullanıcı dostu bir arayüz ve modern bir tasarımla bilimsel verileri kolayca yorumlamanızı sağlar.

#### Uygulama Özellikleri

#### Kullanıcı Dostu Arayüz:
Kullanıcılar, giriş parametrelerini kolayca sağlayarak tahmin alabilirler.
#### Makine Öğrenimi Entegrasyonu:
Random Forest algoritması ile suyun içilebilirliği tahmin edilir.
#### Veri Temizleme:
Eksik değerler sütun medyanları ile doldurularak model performansı optimize edilir.
#### Şık Tasarım:
Gradient arka plan, özelleştirilmiş yazı tipleri ve buton stili ile estetik bir görünüm sunar.

#### Kurulum ve Çalıştırma
Gerekli Kütüphaneler
Uygulamayı çalıştırmadan önce aşağıdaki Python kütüphanelerini yükleyin:
pip install streamlit pandas numpy scikit-learn

#### Başlatma

Uygulamayı başlatmak için aşağıdaki adımları izleyin:
app.py dosyasını bir metin editörü veya IDE ile açın.
Veri yolunu (data_path) kendi dosya yapınıza uygun şekilde değiştirin. Varsayılan yol:
C:\Users\USER\Desktop\Miuul Proje\water_potability.csv

#### Uygulamayı başlatın:
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

#### İnteraktif Arayüz:
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

#### Kullanıcı girişleri
pH = st.slider("pH Değeri", 0.0, 14.0, 7.0)
hardness = st.number_input("Sertlik (mg/L)")
chloramines = st.number_input("Kloramin (mg/L)")

if st.button("Tahmin Yap"):
    prediction = model.predict([[pH, hardness, chloramines]])[0]
    if prediction == 1:
        st.success("Tahmin: İçilebilir 🟢")
    else:
        st.error("Tahmin: İçilemez 🔴")
#### Notlar
Veri yolu (data_path) doğru bir şekilde ayarlanmalıdır.
Veri setinde eksik veri varsa, bu eksiklikler otomatik olarak doldurulacaktır.
