import tensorflow as tf
from keras.datasets import cifar10
#from keras.layers import Dense
#from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import random as rand
from sklearn.metrics import accuracy_score

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth if GPU is detected
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPUs found and configured")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs found")

TRAIN_SIZE = 50000
TEST_SIZE = 10000

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

class_dict = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}

X_train = X_train.reshape(-1, 32*32*3)
X_test = X_test.reshape(-1, 32*32*3)
X_train = X_train.astype('float64') / 255
X_test = X_test.astype('float64') / 255

y_test = np.reshape(y_test, TEST_SIZE)
y_train = np.reshape(y_train, TRAIN_SIZE)

y_train = tf.one_hot(tf.cast(y_train, dtype=tf.int32), 10)
y_test = tf.one_hot(tf.cast(y_test, dtype=tf.int32), 10)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(100, activation="relu", input_shape=(32*32*3,), bias_initializer = "zeros"))
#model.add(tf.keras.layers.Dense(50, activation="relu"))
#model.add(tf.keras.layers.Dense(50, activation="relu"))
#model.add(tf.keras.layers.Dense(25, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

adam = tf.keras.optimizers.SGD(learning_rate=0.001)

model.compile(loss="categorical_crossentropy", optimizer=adam, metrics="accuracy")

model.fit(x=X_train, y=y_train, batch_size=128, epochs=100, shuffle=True)

predictions = model.predict(X_test)

y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(predictions, axis=1)

print(f"accuracy score of test examples is: {accuracy_score(y_test_labels, y_pred_labels)}")

for i in range(10):
    rand.seed()
    rand_index = rand.randint(0, TEST_SIZE)
    img = X_test[rand_index].reshape(32, 32, 3)
    plt.imshow(img)
    plt.title(f"image is a: {class_dict.get(int(tf.argmax(y_test[rand_index]).numpy()))}, "
              f"model predicted:{class_dict.get(predictions[rand_index].argmax())}")
    plt.show()
