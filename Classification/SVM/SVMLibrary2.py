# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 16:29:05 2021

@author: NUGI
"""

#1. Import library
#2. Siapkan dataset(import, clean, prepros)
#3. Split dataset
#4. Bila datanya text maka vektorkan terlebih dahulu, lalu gunakan library classification
#5. Buat report dari hasil model tadi berupa prediksi dan akurasi

#Langkah 1(Import library)
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score

#Langkah 2(Import dataset, clean)
df = pd.read_csv('../Dataset/Spam.csv', encoding="latin-1")
df.rename(columns={'v1':'Label', 'v2':'Text'}, inplace= True)
df.drop(df.columns[[2,3,4]], axis=1, inplace=True)
# print(df.head())
# print(df.describe())

#Langkah 3(Split data menjadi X dan Y lalu displit lagi menjadi data train dan data test)
X = df['Text'].values
y = df['Label'].values

bin = LabelBinarizer()
y = bin.fit_transform(y).ravel()
# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Langkah 4(Gunakan library untuk mengvektorkan data, dan fit data train)
vek = TfidfVectorizer(stop_words='english')
X_train_vek = vek.fit_transform(X_train)
X_test_vek = vek.transform(X_test)
# print(X_train_vek)

model = svm.SVC(kernel='linear')
print(model.fit(X_train_vek, y_train)) 

#Langkah 5(Buat prediksi dan akurasi lalu cetak, jangan lupa gunakan fitur report klasifikasi)
prediksi = model.predict(X_test_vek)
akurasi = accuracy_score(y_test, prediksi)
print(prediksi)
print(akurasi)
print(classification_report(y_test, prediksi))