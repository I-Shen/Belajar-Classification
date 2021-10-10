# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:22:31 2021

@author: NUGI
"""

#1. Import library
#2. Siapkan dataset(import, clean, prepros)
#3. Split dataset
#4. Gunakan library classification
#5. Buat report dari hasil model tadi berupa prediksi dan akurasi

#Langkah 1(Import library)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

#Langkah 2(Import dataset)
df = pd.read_csv('../Dataset/Iris.csv')
# print(df.head())
# print(df.describe())

#Langkah 3(Split data menjadi X dan Y lalu displit lagi menjadi data train dan data test)
X = df.iloc[:, [0,1,2,3]].values
y = df.iloc[:, 4].values
# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Langkah 4(Gunakan library dan fit data train)
model = RandomForestClassifier()
print(model.fit(X_train, y_train))

#Langkah 5(Buat prediksi dan akurasi lalu cetak, jangan lupa gunakan fitur report klasifikasi)
prediksi = model.predict(X_test)
akurasi = accuracy_score(y_test, prediksi)
print(prediksi)
print(akurasi)
print(classification_report(y_test, prediksi))
