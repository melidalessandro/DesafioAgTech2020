# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 19:56:16 2020

@author: mdalessandro
"""

import zipfile
import keras
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

path_to_zip_file = "dataset/raw/df_train_rn_full_v2.zip"

with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall('dataset/raw/')
    
# Carga del dataset
dataset_path = "dataset/raw/df_train_rn_full_v2.csv"
data=np.loadtxt(open(dataset_path, "rb"), delimiter=",", skiprows=1)
x,y=data[:,2:524] ,data[:,525]
# cantidad de ejemplos y dimension de entrada
n,d_in=x.shape
# calcula la cantidad de clases
classes=int(y.max()+1)

print("Información del conjunto de datos:")
print(f"Ejemplos: {n}")
print(f"Variables de entrada: {d_in}")
print(f"Cantidad de clases: {classes}")

# Veo si hay desbalance de clases
data['CultivoId'].value_counts()
# Hay desbalance
list(data)

# Separación en train y test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Normalizado de las variables numéricas de train y test por separado
for i in range(d_in): 
    x_train[:,i]=(x_train[:,i]-x_train[:,i].mean())/x_train[:,i].std()
for i in range(d_in): #
    x_test[:,i]=(x_test[:,i]-x_test[:,i].mean())/x_test[:,i].std()

# Modelo Red Neuronal
modelo = keras.Sequential([
    keras.layers.Dense(256,input_shape=(d_in,), activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(classes, activation='softmax')])

modelo.compile(
  optimizer=keras.optimizers.SGD(lr=0.001), 
  loss='sparse_categorical_crossentropy', 
  metrics=['accuracy'],
)

# Validación
epocas=100
history = modelo.fit(x_train, y_train, epochs=epocas, batch_size=16, validation_data=(x_test,y_test), class_weight='balanced')

y_pred = modelo.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis = 1)

# Métricas
metrics.cohen_kappa_score(y_test, y_pred_labels, labels=None, weights=None, sample_weight=None)
metrics.balanced_accuracy_score(y_test, y_pred_labels)

# Entrenamiento modelo completo
# # Normalizado de las variables numéricas del modelo para train
for i in range(d_in):
    x[:,i]=(x[:,i]-x[:,i].mean())/x[:,i].std()
    
# Entrenamiento del modelo completo
modelo.fit(x, y, epochs=epocas, batch_size=16, class_weight='balanced')

# Guardo el modelo
modelo.save('code/RN/RN_v2/modelo_v2.h5')
