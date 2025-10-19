import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Load and Preprocess Data
print("Loading and preprocessing data...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape data for CNN input (batch_size, height, width, channels)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# Normalize pixel values to the range [0, 1]
X_train /= 255
X_test /= 255

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 2. Build the CNN Model
print("Building the CNN model...")
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 3. Train the Model
print("\nTraining the model...")
model.fit(X_train, y_train,
          epochs=10,
          batch_size=200,
          validation_data=(X_test, y_test),
          verbose=1)

# 4. Evaluate the Model
print("\nEvaluating the model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

# 5. Make a Prediction and Visualize
print("\nMaking a prediction on a sample image...")
sample_index = 0
sample_image = X_test[sample_index]
true_label = np.argmax(y_test[sample_index])
image_for_prediction = np.expand_dims(sample_image, axis=0)
prediction = model.predict(image_for_prediction)
predicted_label = np.argmax(prediction)

print(f"True Label: {true_label}")
print(f"Predicted Label: {predicted_label}")

plt.imshow(sample_image.reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predicted_label}, True: {true_label}")
plt.axis('off')
plt.show()
