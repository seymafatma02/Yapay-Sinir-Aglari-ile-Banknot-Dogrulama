import pandas as pd

# Veri yolunu belirle
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"

# Veriyi yükle
columns = ["Variance", "Skewness", "Kurtosis", "Entropy", "Class"]
data = pd.read_csv(data_url, header=None, names=columns)

# CSV olarak kaydet
data.to_csv("banknote_authentication.csv", index=False)

# Alternatif olarak Excel dosyasına yazdır
data.to_excel("banknote_authentication.xlsx", index=False)

print("Veri başarıyla Excel'e yazdırıldı!")
