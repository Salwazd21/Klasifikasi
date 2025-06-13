🌱 Image Classification with CNN - Plant Seedlings Dataset
Proyek ini merupakan bagian dari submission pelatihan deep learning untuk klasifikasi gambar tanaman berdasarkan dataset Plant Seedlings. Model yang digunakan adalah CNN (Convolutional Neural Network) dan telah dikonversi ke format TensorFlow Lite dan TensorFlow.js.

📁 Struktur Folder
Salin
Edit
Split_Dataset/
├── Train/
├── Val/
└── Test/
    ├── Maize/
    ├── Fat Hen/
    └── ... (12 kelas total)
📌 1. Library
Notebook ini menggunakan pustaka berikut:

python
Salin
Edit
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
📥 2. Load Dataset
Dataset dibagi menjadi 3 folder: Train, Validation, dan Test. Semua gambar di-rescale ke ukuran 150x150.

📊 3. Split Dataset
Dataset telah dipisah secara manual menjadi 3 bagian:

Train: Untuk pelatihan model.

Validation: Untuk menghindari overfitting.

Test: Untuk evaluasi akhir dan inference.

🧠 4. CNN Model
Model CNN terdiri dari beberapa lapisan konvolusi, max-pooling, dropout, dan softmax di akhir. Kompilasi menggunakan categorical_crossentropy dan optimizer adam.

📈 5. Evaluasi & Visualisasi
Model dievaluasi menggunakan akurasi validasi dan visualisasi menggunakan matplotlib. Juga tersedia:

Confusion Matrix

Grafik Loss & Accuracy

📦 6. Konversi Model
Model berhasil dikonversi ke:

.tflite (TensorFlow Lite)

tfjs (TensorFlow.js)

bash
Salin
Edit
# Save Model
model.export("saved_model")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
tflite_model = converter.convert()

# Convert to TFJS
!tensorflowjs_converter --input_format=tf_saved_model saved_model tfjs_model
🔍 7. Inference
Prediksi dilakukan dengan memuat gambar dari folder Split_Dataset/Test/<NamaKelas>/<file>.jpg dan hasil ditampilkan secara visual.

python
Salin
Edit
predict_image("Maize", "img_42.jpg", model, train_gen)
