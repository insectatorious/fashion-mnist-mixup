import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from model import get_model
from utils import mix_up

BATCH_SIZE = 32
fashion_mnist = tf.keras.datasets.fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.astype(np.float64) / 255.
X_test = X_test.astype(np.float64) / 255.

X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=0.1,
                                                  stratify=y_train)
y_val = tf.one_hot(y_val, 10, dtype=tf.float64)
y_test = tf.one_hot(y_test, 10, dtype=tf.float64)
y_train = tf.one_hot(y_train, 10, dtype=tf.float64)

train_set_one = (tf.data.Dataset
                 .from_tensor_slices((X_train, y_train))
                 .shuffle(BATCH_SIZE * 100)
                 .batch(BATCH_SIZE))
train_set_two = (tf.data.Dataset
                 .from_tensor_slices((X_train, y_train))
                 .shuffle(BATCH_SIZE * 100)
                 .batch(BATCH_SIZE))
train_data = tf.data.Dataset.zip((train_set_one, train_set_two))
val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

train_data_mu = train_data.map(lambda one, two: mix_up(one, two, alpha=5.1),
                               num_parallel_calls=tf.data.AUTOTUNE)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model: tf.keras.Model = get_model(pad_inputs=False)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])

model.summary()

logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                                      update_freq="batch",
                                                      histogram_freq=10)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="/tmp/model",
                                                 save_weights_only=True,
                                                 save_best_only=True)
model.fit(train_data_mu, validation_data=val_data, epochs=20,
          callbacks=[tensorboard_callback, cp_callback])
model.load_weights("/tmp/model")
model.evaluate(test_data)
