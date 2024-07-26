import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# Load the Food-101 dataset
dataset, info = tfds.load('food101', with_info=True, as_supervised=True)
train_data, test_data = dataset['train'], dataset['validation']

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
TRAIN_SAMPLES = 5000  # Number of training samples to use
TEST_SAMPLES = 1000   # Number of testing samples to use

# Data preprocessing function
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Take a subset of the data
train_data = train_data.take(TRAIN_SAMPLES)
test_data = test_data.take(TEST_SAMPLES)

# Preprocess the datasets
train_data = train_data.map(preprocess).cache().shuffle(1000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
test_data = test_data.map(preprocess).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# Load the pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                               include_top=False,
                                               weights='imagenet')

# Freeze the base model
base_model.trainable = False

# Create the model
model = tf.keras.Sequential([
    data_augmentation,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(info.features['label'].num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_data, validation_data=test_data, epochs=10)

# Unfreeze some layers of the base model for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

# Recompile the model
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fine-tune the model
fine_tune_epochs = 10
total_epochs = 10 + fine_tune_epochs
history_fine = model.fit(train_data, validation_data=test_data, epochs=total_epochs, initial_epoch=history.epoch[-1])

# Evaluate the model
loss, accuracy = model.evaluate(test_data)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Detailed evaluation metrics
from sklearn.metrics import classification_report

def get_classification_report(data, model):
    y_true = []
    y_pred = []

    for images, labels in data:
        predictions = model.predict(images)
        pred_labels = np.argmax(predictions, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(pred_labels)

    report = classification_report(y_true, y_pred, target_names=info.features['label'].names)
    return report

report = get_classification_report(test_data, model)
print(report)

# Visualize some predictions
def plot_predictions(data, model, num_images=5):
    plt.figure(figsize=(15, 15))
    for images, labels in data.take(1):
        predictions = model.predict(images)
        for i in range(num_images):
            ax = plt.subplot(1, num_images, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            pred_label = np.argmax(predictions[i])
            true_label = labels[i].numpy()
            plt.title(f"True: {info.features['label'].names[true_label]}, Pred: {info.features['label'].names[pred_label]}")
            plt.axis("off")

plot_predictions(test_data, model)

# Visualize misclassified images
def plot_misclassified(data, model, num_images=5):
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    for images, labels in data.take(10):
        predictions = model.predict(images)
        pred_labels = np.argmax(predictions, axis=1)
        for i in range(len(labels)):
            if labels[i].numpy() != pred_labels[i]:
                misclassified_images.append(images[i].numpy())
                misclassified_labels.append(labels[i].numpy())
                misclassified_preds.append(pred_labels[i])
            if len(misclassified_images) >= num_images:
                break
        if len(misclassified_images) >= num_images:
            break

    plt.figure(figsize=(15, 15))
    for i in range(num_images):
        ax = plt.subplot(1, num_images, i + 1)
        plt.imshow(misclassified_images[i].astype("uint8"))
        plt.title(f"True: {info.features['label'].names[misclassified_labels[i]]}, Pred: {info.features['label'].names[misclassified_preds[i]]}")
        plt.axis("off")

plot_misclassified(test_data, model)
