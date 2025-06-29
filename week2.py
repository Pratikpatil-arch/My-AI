# === 1. Install TensorFlow (only needed if not already installed) ===
# !pip install tensorflow

# === 2. Import Libraries ===
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet import preprocess_input

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import gradio as gr
from PIL import Image
import os

# === 3. Set Dataset Paths ===
trainpath = r'C:\Users\Edunet Foundation\Downloads\project\E waste data\modified-dataset\train'
validpath = r'C:\Users\Edunet Foundation\Downloads\project\E waste data\modified-dataset\val'
testpath  = r'C:\Users\Edunet Foundation\Downloads\project\E waste data\modified-dataset\test'

# === 4. Load Datasets ===
datatrain = tf.keras.utils.image_dataset_from_directory(
    trainpath, shuffle=True, image_size=(128, 128), batch_size=32)

datavalid = tf.keras.utils.image_dataset_from_directory(
    validpath, shuffle=True, image_size=(128, 128), batch_size=32)

datatest = tf.keras.utils.image_dataset_from_directory(
    testpath, shuffle=False, image_size=(128, 128), batch_size=32)

# === 5. Class Names ===
class_names = datatrain.class_names
print("Classes:", class_names)

# === 6. Prefetch for performance ===
AUTOTUNE = tf.data.AUTOTUNE
datatrain = datatrain.prefetch(AUTOTUNE)
datavalid = datavalid.prefetch(AUTOTUNE)
datatest  = datatest.prefetch(AUTOTUNE)

# === 7. Build Model ===
base_model = EfficientNetV2B0(include_top=False, input_shape=(128, 128, 3), weights='imagenet')
base_model.trainable = False

model = Sequential([
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

# === 8. Compile Model ===
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# === 9. Train Model ===
history = model.fit(datatrain, validation_data=datavalid, epochs=10)

# === 10. Save Model ===
model.save("e_waste_model.h5")
print("Model saved as 'e_waste_model.h5'")

# === 11. Plot Accuracy & Loss ===
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Accuracy")
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss")
plt.show()

# === 12. Evaluate on Test Set ===
model.evaluate(datatest)

# === 13. Predictions & Evaluation ===
predictions = model.predict(datatest)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.concatenate([labels for _, labels in datatest], axis=0)

conf_mat = confusion_matrix(true_labels, predicted_labels)
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:\n")
print(classification_report(true_labels, predicted_labels, target_names=class_names))

# === 14. Gradio App to Predict Images ===
def predict_image(img):
    img = img.resize((128, 128))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    return {class_names[i]: float(pred[0][i]) for i in range(len(class_names))}

interface = gr.Interface(fn=predict_image,
                         inputs=gr.Image(type="pil"),
                         outputs=gr.Label(num_top_classes=3),
                         title="E-Waste Classifier")
interface.launch()
