# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 19:56:16 2020

@author: mdalessandro
"""
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from sklearn.utils import class_weight
import zipfile


# Carga del dataset
path_to_zip_file = "dataset/raw/df_rn_base.zip"

dataset_path = "dataset/raw/df_rn_base.csv"

with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall('dataset/raw/')
    
data = pd.read_csv(dataset_path)
data.head()
# Veo si hay desbalance de clases
data['CultivoId'].value_counts()
# Hay desbalance pero no es terrible
import tensorflow as tf
import keras
import numpy as np
import os

# Carga del dataset
#dataset_path=os.path.join("../../",dataset)
data=np.loadtxt(open(dataset_path, "rb"), delimiter=",", skiprows=1)
x,y=data[:,2:324] ,data[:,0]
# cantidad de ejemplos y dimension de entrada
n,d_in=x.shape
# calcula la cantidad de clases
classes=int(y.max()+1)


print("Información del conjunto de datos:")
print(f"Ejemplos: {n}")
print(f"Variables de entrada: {d_in}")
print(f"Cantidad de clases: {classes}")

# Separacion en train y test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
x_train
y_train


# Normalizo las variables de entrada
for i in range(d_in): 
    x_train[:,i]=(x_train[:,i]-x_train[:,i].mean())/x_train[:,i].std()

# Normalizo las variables de salida
for i in range(d_in): #
    x_test[:,i]=(x_test[:,i]-x_test[:,i].mean())/x_test[:,i].std()

# Creación del modelo inicial
print("Inicialización aleatoria del modelo (podes volver a correr esta celda para obtener otros resultados)")

# Creo un modelo Red Neuronal 
modelo = keras.Sequential([
    # input_shape solo en la primer capa
    # Capa con 3 salidas, activación relu
    keras.layers.Dense(256,input_shape=(d_in,), activation='relu'),
    # Capa con 5 salidas, activación tanh
    keras.layers.Dense(256, activation='relu'),
    # Capa con 5 salidas, activación tanh
    #keras.layers.Dense(48, activation='tanh'),
    #la ultima capa si o si tiene que tener tantas salidas como clases, y softmax 
    keras.layers.Dense(classes, activation='softmax')])


modelo.compile(
  optimizer=keras.optimizers.Adam(lr=0.001), 
  loss='sparse_categorical_crossentropy', 
  # metricas para ir calculando en cada iteracion o batch 
  # Agregamos el accuracy del modelo
  metrics=['accuracy'], 
)

# Entrenamiento del modelo
epocas=100
history = modelo.fit(x_train,y_train,epochs=epocas,batch_size=16, validation_data = (x_test,y_test), class_weight='balanced')


y_pred = modelo.predict(x_test)
y_pred_labels = np.argmax(y_pred,axis = 1)

metrics.cohen_kappa_score(y_test, y_pred_labels, labels=None, weights=None, sample_weight=None)
metrics.balanced_accuracy_score(y_test, y_pred_labels)

# Entrenamiento modelo completo
# Normalizo las variables de entrada del modelo total
for i in range(d_in):
    x[:,i]=(x[:,i]-x[:,i].mean())/x[:,i].std()
    
# Entrenamiento del modelo
modelo.fit(x,y,epochs=50,batch_size=16, class_weight='balanced')

#salvo el modelo
modelo.save('code/modelo_base/modelo_base.h5')