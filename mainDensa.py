from pickletools import optimize
import numpy as np
from sklearn import metrics
import tensorflow as tf
import os
import re
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import keras
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

# INCIA LA LECTURA DE LAS IMAGENES Y EL DATASET DESCARGAOD DE INTERNET
dirname = os.path.join(os.getcwd(), "deportes")
imgpath = dirname + os.sep

images = []
directories = []
dircount = []
prevRoot = ""
cant = 0

print("leyendo imagenes de ", imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            cant = cant + 1
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            images.append(image)
            b = "Leyendo..." + str(cant)
            print(b, end="\r")
            if prevRoot != root:
                print(root, cant)
                prevRoot = root
                directories.append(root)
                dircount.append(cant)
                cant = 0
dircount.append(cant)
dircount = dircount[1:]
dircount[0] = dircount[0] + 1

print("Directorios leidos:", len(directories))
print("Imagenes en cada directorio", dircount)
print("suma Total de imagenes en subdirs:", sum(dircount))

print("\n")
print(f" cuantas categorias imagenes tenemos {len((images))}")
print(f"cuantas imagenes{len((images[0]))}")
print(f"cuantas filas{len((images[0][0]))}")
print(f"cuantas columnas{len((images[0][0][0]))}")

print("\n")
print(f"dircount {dircount}")

# creacion de etiquetas para la identifiacion de las imagenes y deportes

labels = []
indice = 0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice = indice + 1

print("Cantidad etiquetas creadas: ", len(labels))


deportes = []
indice = 0
for directorio in directories:
    name = directorio.split(os.sep)
    print(indice, name[len(name) - 1])
    deportes.append(name[len(name) - 1])
    indice = indice + 1

print("\n")
print(f"deportes {deportes}")

y = np.array(labels)
X = np.array(images, dtype=np.uint8)  # convierto de lista a numpy

print("\n")
print(f"salida {len(y)}")
print(f"entrada {len(X)}")

# Find the unique numbers from the train labels
classes = np.unique(y)
nClasses = len(classes)
print("numero de clases : ", nClasses)
print("clases: ", classes)


# Mezclar todo y crear los grupos de entrenamiento y testing
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.1)

print("entrenamiento : ", train_X.shape, train_Y.shape)
print("test : ", test_X.shape, test_Y.shape)

train_X = train_X.astype("float32")
test_X = test_X.astype("float32")

# NORMALIZACION DE DATOS

print(f"conversion x a 32: {train_X.shape}")
print(f"conversion y a 32: {train_Y.shape}")
train_X = train_X / 255.0
test_X = test_X / 255.0
print("\n")
print(f"conversion x a 32 con division: {train_X.shape}")
print(f"conversion y a 32 con division: {train_Y.shape}")

train_Y_one_hot = to_categorical(train_Y)
print(f"conversion x a categorical: {train_Y_one_hot.shape}")


print(train_X.shape, test_Y.shape)

# # tasa de aprendizaje
n = 0.0001
INIT_LR = 1e-3

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(21, 28, 3)),
        tf.keras.layers.Dense(50, activation=tf.nn.relu),
        tf.keras.layers.Dense(50, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.sigmoid),
    ]
)


model.compile(
    optimizer=tf.keras.optimizers.Adagrad(
        lr=INIT_LR, epsilon=None, decay=INIT_LR / 100
    ),
    metrics=["accuracy"],
    loss='categorical_crossentropy',
)
# #
# # Train the perceptron using stochastic gradient descent
# # with a validation split of 20%
historial = model.fit(train_X, train_Y_one_hot, batch_size=64,epochs=10,verbose=1)
