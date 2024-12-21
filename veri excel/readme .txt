# Yapay Sinir Ağları ile Banknot Doğrulama  

## Genel Bakış  
Bu proje, banknotların gerçek mi yoksa sahte mi olduğunu sınıflandırmayı amaçlamaktadır. Banknotlardan elde edilen varyans, çarpıklık, basıklık ve entropi gibi özellikler kullanılarak bir yapay sinir ağı modeli eğitilmiştir. Proje, veri ön işleme, model eğitimi ve değerlendirme adımlarını kapsamaktadır.  

## Özellikler  
- **Giriş Özellikleri**:  
  - Varyans  
  - Çarpıklık  
  - Basıklık  
  - Entropi  
- **Çıkış**: İkili sınıflandırma (0: Sahte, 1: Gerçek)  
- **Ön İşleme**: Veriler `StandardScaler` kullanılarak standartlaştırılmıştır.  
- **Model Mimarisi**:  
  - Üç gizli katman (128, 64, 32 nöronlu tam bağlı katmanlar)  
  - Aşırı öğrenmeyi önlemek için Dropout katmanları  
  - Çıkış katmanında sigmoid aktivasyon fonksiyonu  

## Veri Kümesi  
Kullanılan veri kümesi "Banknote Authentication" veri kümesidir. Veri kümesinde 5 sütun bulunmaktadır:  
1. Varyans  
2. Çarpıklık  
3. Basıklık  
4. Entropi  
5. Sınıf (0 veya 1)  

## Proje Yapısı  
- `banknote_authentication.py`: Yapay sinir ağının eğitim ve test işlemlerini gerçekleştiren ana script.  
- `banknote_authentication.xlsx`: Veri kümesi dosyası.  
- `README.md`: Proje dokümantasyonu.  

## Gereksinimler  
Bu projeyi çalıştırmak için aşağıdaki kütüphaneler gereklidir:  
- Python 3.x  
- TensorFlow  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Seaborn  

Gerekli kütüphaneleri şu komutla yükleyebilirsiniz:  
```bash
pip install -r requirements.txt
