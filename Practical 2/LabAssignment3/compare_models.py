import tensorflow as tf
from tensorflow.keras import layers, models

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test  = x_test / 255.0

# Dense model
dense = models.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

dense.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

dense.fit(x_train, y_train, epochs=5, verbose=0)
dense_acc = dense.evaluate(x_test, y_test, verbose=0)[1]

# CNN model
cnn = models.Sequential([
    layers.Reshape((28,28,1), input_shape=(28,28)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cnn.fit(x_train, y_train, epochs=5, verbose=0)
cnn_acc = cnn.evaluate(x_test, y_test, verbose=0)[1]

print("Dense accuracy:", dense_acc)
print("CNN accuracy:", cnn_acc)
