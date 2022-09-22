# %%
## Importación de Librerías ##
# Se usan librerías para lectura de archivos, funciones matemáticas, graficar y de redes neuronales.
import numpy as np
import pandas as pd
import os
import tensorflow as tf 
from tensorflow import keras
from keras.models import Sequential
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing import image 
from tensorflow.keras.preprocessing.image import ImageDataGenerator , load_img ,img_to_array
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix

# %%
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %%

## Lectura de datos
train_path = 'C:/Users/Panda/Downloads/casting_data/train/'
test_path = 'C:/Users/Panda/Downloads/casting_data/test/'
mixed_train = 'C:/Users/Panda/Downloads/casting_data/personalized_dataset/mixed_dataset/train/'
mixed_test = 'C:/Users/Panda/Downloads/casting_data/personalized_dataset/mixed_dataset/test/'

# Se declaran el tamaño de las imagenes, y cual es el batch size a usar (se deja el predeterminado de Keras)
image_shape = (300,300,1)
batch_size = 32

# Se imprimen la cantidad de elementos por categoria 
print("Class ok_front train count:",len(os.listdir(train_path+'ok_front')))
print("Class def_front train count:",len(os.listdir(train_path+'def_front')) )

print("Class def_front test count:",len(os.listdir(test_path+'def_front')))
print("Class ok_front test count:",len(os.listdir(test_path+'ok_front')))

# Se normalizan las imagenes en una escala de 0-1
image_gen = ImageDataGenerator(rescale=1/255)# Rescale the image by normalzing it)

# Se generan los batches de imagenes, utilizando el tamaño original, en un modo de escalas de gris y clasificandolo de manera binaria-
train_set = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2], # Target size (300x300)
                                               color_mode="grayscale",
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=True) #Shuffle false would sorts the data in alphanumeric order.

test_set = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode="grayscale",
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)
mixed_train_set = image_gen.flow_from_directory(mixed_train,
                                               target_size=image_shape[:2], # Target size (300x300)
                                               color_mode="grayscale",
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=True) #Shuffle false would sorts the data in alphanumeric order.

mixed_test_set = image_gen.flow_from_directory(mixed_test,
                                               target_size=image_shape[:2],
                                               color_mode="grayscale",
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)

# Se crean variables con los valores de clases, tambien se imprimen para mejor visualización
class_labels = train_set.class_indices
print(class_labels)
classes = list(class_labels.keys())
print(classes)

# %%

## Funcion para imprimir el primer batch de imagenes para mejor visualización
def visualizeImageBatch(dataset, title):
    mapping_class = {1: "Okay", 0: "Defectuoso"}
    images, labels = next(iter(dataset))
    images = images.reshape(batch_size, *(image_shape[:2]))
    fig, axes = plt.subplots(4, 8, figsize=(16,10))  # Se tiene que cambiar este valor dependiendo del tamaño del batch

    for ax, img, label in zip(axes.flat, images, labels):
        ax.imshow(img, cmap = "gray")
        ax.axis("off")
        ax.set_title(mapping_class[label], size = 16)

    plt.tight_layout()
    fig.suptitle(title, size = 30, y = 1.05, fontweight = "bold")
    plt.show()
    
    return images

train_images = visualizeImageBatch(train_set,"Primeros datos de Entrenamiento")

# %%

## Creación de la red neuronal convolutiva
cnn_model = tf.keras.Sequential()

# Primero se hacen 3 capas conlutivas con pooling
cnn_model.add(Conv2D(filters=8, kernel_size=(3,3),input_shape=image_shape, activation='relu', strides=(1,1)))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Conv2D(filters=16, kernel_size=(3,3),input_shape=image_shape, activation='relu', strides=(1,1)))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
# Se aplica dropout, la cual de manera aleatoria pone inputs en 0 con una frecuencia de 0.25 para prevenir sobreentrenamiento.
cnn_model.add(Dropout(0.25))
cnn_model.add(Conv2D(filters=16, kernel_size=(3,3),input_shape=image_shape, activation='relu', strides=(1,1)))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten
cnn_model.add(Flatten())

# Se termina con una red densa clasica con una neurona de salida
cnn_model.add(Dense(units=64,activation='sigmoid')) 
cnn_model.add(Dense(units=1,activation='sigmoid'))

# Se compila utilizando Optimizador de Adam, perdida de binary_cross entropy
cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])

# Se imprime un resumen del modelo.
cnn_model.summary()

# %%
## Entrenamiento

# EarlyStopping es un metodo para detener el entrenamiento cuando la metrica que interesa deja de mejorar.
# En nuestro caso elegimos la perdida de validacion y la paciencia es cuantas epocas tienen que pasar sin mejora antes de detenerse.
early_stop = EarlyStopping(monitor='loss',patience=2)

# Se hace el entrenamiento
cnn_model.fit_generator(train_set,validation_data=test_set,epochs=10,callbacks=early_stop)

# Entrenamiento de datos mixed
#cnn_model.fit_generator(mixed_train_set,validation_data=mixed_test_set,epochs=10,callbacks=early_stop)

# %%

# Se grafican los valores de perdida/precisión de entrenamiento y validación
losses = pd.DataFrame(cnn_model.history.history)
losses[['loss','val_loss']].plot()

losses[['accuracy','val_accuracy']].plot()

# Se utiliza el set de test para predicción
pred_probability = cnn_model.predict_generator(test_set)
#pred_probability = cnn_model.predict_generator(mixed_test_set)

#test_set.classes

# Si el modelo predice mas de 0.5 se convierte a 1 (OK_front)
predictions = pred_probability > 0.5

# Se imprime el reporte de calificación
print(classification_report(test_set.classes,predictions))
#print(classification_report(mixed_test_set.classes,predictions))

# %%

## Finalmente se genera la matriz de confusión y se grafica.
plt.figure(figsize=(8,8))
cnf_matrix = confusion_matrix(test_set.classes,predictions)
#cnf_matrix = confusion_matrix(mixed_test_set.classes,predictions)

sns.heatmap(cnf_matrix, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class", fontsize=18)
plt.ylabel("True Class", fontsize=18)
class_count=len(classes)
nice_classes = ['Defective', 'Okay']
plt.xticks(np.arange(class_count)+.5, nice_classes)
plt.yticks(np.arange(class_count)+.5, nice_classes)

# %%
## Separando datos
# Solo se necesita correr una vez cuando se tienen los datos mezclados
import splitfolders

mixed_dir_def = 'C:/Users/Panda/Downloads/casting_data/personalized_dataset/train_personalized_def'
mixed_dir_ok = 'C:/Users/Panda/Downloads/casting_data/personalized_dataset/train_personalized_ok'
mixed_dataset = 'C:/Users/Panda/Downloads/casting_data/personalized_dataset/mixed_dataset'

# Deffective
splitfolders.ratio(mixed_dir_def, output=mixed_dataset,
    seed=1337, ratio=(0.89,0.11), move=False)

#Okay
splitfolders.ratio(mixed_dir_ok, output=mixed_dataset,
    seed=1337, ratio=(0.916,0.084), move=False)
# %%
