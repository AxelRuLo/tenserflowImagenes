
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# INCIA LA LECTURA DE LAS IMAGENES Y EL DATASET DESCARGAOD DE INTERNET
def densa2(images, directories, dircount, prevRoot, cant):

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
            tf.keras.layers.Dense(50, activation=tf.nn.relu),
            tf.keras.layers.Dense(50, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adagrad(
            lr=INIT_LR, epsilon=None, decay=INIT_LR / 100
        ),
        metrics=["accuracy"],
        loss="categorical_crossentropy",
    )
    # #
    # # Train the perceptron using stochastic gradient descent
    # # with a validation split of 20%
    historial = model.fit(train_X, train_Y_one_hot, batch_size=64, epochs=10, verbose=1)

    return historial.history["loss"]
