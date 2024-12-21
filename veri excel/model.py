import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt 
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras import optimizers # type: ignore
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Excel dosyasının yolunu belirtin
data_path = "C:/veri excel/banknote_authentication.xlsx"

# Veriyi yükle
data = pd.read_excel(data_path)

# Veriyi kontrol et
print(data.head())

# Model için veriyi ayırma
X = data[["Variance", "Skewness", "Kurtosis", "Entropy"]]
y = data["Class"]


# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellikleri standartlaştırma (veri aralıklarını 0-1 arasına çekmek için)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# YSA Modelini Kurma
model = Sequential()

# Giriş katmanı ve ilk gizli katman (4 giriş özelliği, nöronlu gizli katman)



model.add(Dense(128, input_dim=4, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, input_dim=4, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, input_dim=4, activation='relu'))
# Dropout katmanı ekleyerek aşırı öğrenmeyi engelleme





# Çıkış katmanı (binary sınıflandırma için 1 nöronlu çıkış)
model.add(Dense(1, activation='sigmoid'))

# Modeli derleme
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(learning_rate=0.00001), metrics=['accuracy'])

# Erken durdurma (overfitting'i engellemek için)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Modeli eğitme
# Sınıf ağırlıklarını hesapla
y_train_np = y_train.to_numpy()  # Pandas Series'i numpy dizisine dönüştür
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_np), y=y_train_np)
class_weights = dict(enumerate(class_weights))
history =model.fit(X_train, y_train, epochs=250, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Model Performansını Görselleştir
plt.figure(figsize=(4,4))
plt.plot(history.history['accuracy'])  # 'acc' yerine 'accuracy' kullanın
plt.plot(history.history['val_accuracy'])  # 'val_acc' yerine 'val_accuracy' kullanın
plt.title("Model Doğruluk Oranı")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.savefig('sema.png', dpi=300)

print(f"En iyi validation accuracy: {max(history.history['val_accuracy'])}")

# Modeli değerlendirme (test verisi üzerinde)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")


# Eğitim tamamlandıktan sonra Train Accuracy ve Train Loss'u ekrana yazdırma
final_train_loss = history.history['loss'][-1]  # Son epoch'taki eğitim kaybı (loss)
final_train_accuracy = history.history['accuracy'][-1]  # Son epoch'taki eğitim doğruluğu (accuracy)

print(f"Train Loss (Eğitim Kaybı): {final_train_loss:.4f}")
print(f"Train Accuracy (Eğitim Doğruluğu): {final_train_accuracy*100:.2f}%")

















