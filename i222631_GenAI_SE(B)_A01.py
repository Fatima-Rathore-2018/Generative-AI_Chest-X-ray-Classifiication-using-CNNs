# Training parameters.
EPOCHS = 20                 # Number of epochs.
BATCH_SIZE = 16             # Batch size.
LEARNING_RATE = 0.0003       # Initial learning rate.

# Regularization.
DROPOUT_RATE = 0.4          # Dropout in Dense layers.
L2_LAMBDA = 0.001           # L2 regularization coefficient.

# Callbacks.
EARLY_STOP_PATIENCE = 5
REDUCE_LR_PATIENCE = 3

# Image parameters.
IMAGE_SIZE = (224, 224)  # Input image size for the model.

# Data Augmentation.
ROTATION_RANGE = 15
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
SHEAR_RANGE = 0.1
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True

print("Hyperparameters defined.")

# from google.colab import drive
# drive.mount('/content/drive')
# print("Google Drive mounted.")

import os
#dataset_path = 'E:/FAST NUCES_BS(SE)/Semester 08 (Spring 2026)/GenAI/Assignment 01/i222631_GenAI_A01/chest_xray'
dataset_path = os.path.join(os.getcwd(), "chest_xray")

if not dataset_path:
    print("Dataset path not specified.")
else:
    print(f"Dataset path: {dataset_path}")

print("Folders in dataset:", os.listdir(dataset_path))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
import pandas as pd

# Define training directories for each class.
train_normal = os.path.join(dataset_path, 'train', 'NORMAL')
train_pneumonia = os.path.join(dataset_path, 'train', 'PNEUMONIA')

# Number of images to display
num_images = 9

# Collect all image file paths for normal and pneumonia
all_normal = [os.path.join(train_normal, f) for f in os.listdir(train_normal)]
all_pneumonia = [os.path.join(train_pneumonia, f) for f in os.listdir(train_pneumonia)]

# Randomly sample images from both classes
sample_images = random.choices(all_normal, k=num_images//2) + random.choices(all_pneumonia, k=num_images - num_images//2)
random.shuffle(sample_images)  # Shuffle to mix classes

# Create 3x3 grid
plt.figure(figsize=(10,10))

for i, img_path in enumerate(sample_images):
    img = mpimg.imread(img_path)
    label = "Normal" if "NORMAL" in img_path else "Pneumonia"
    
    plt.subplot(3, 3, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(label)
    plt.axis('off')

plt.tight_layout()
plt.show()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Training & Validation Generators.
# Create training data generator with augmentation and normalization.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=ROTATION_RANGE,
    width_shift_range=WIDTH_SHIFT_RANGE,
    height_shift_range=HEIGHT_SHIFT_RANGE,
    zoom_range=ZOOM_RANGE,
    horizontal_flip=HORIZONTAL_FLIP,
    fill_mode='nearest',
)

# Load training images from directory and apply preprocessing.
train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_path, 'train'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Create validation generator with only normalization.
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load validation images.
validation_generator = validation_datagen.flow_from_directory(
    os.path.join(dataset_path, 'val'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Create test generator (preserve label order).
test_datagen = ImageDataGenerator(rescale=1./255)

# Load test images.
test_generator = test_datagen.flow_from_directory(
    os.path.join(dataset_path, 'test'),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print("Data generators ready with augmentation and normalization.")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers

# Build custom CNN model using sequential architecture.
model = Sequential([
     # First convolutional block for low-level feature extraction.
    Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPool2D(2,2),
    BatchNormalization(),

    # Second convolutional block for mid-level features.
    Conv2D(64, (3,3), activation='relu'),
    MaxPool2D(2,2),
    BatchNormalization(),

    # Third convolutional block for deeper feature learning.
    Conv2D(128, (3,3), activation='relu'),
    MaxPool2D(2,2),
    BatchNormalization(),

    # Flatten feature maps into a single vector.
    Flatten(),

    # Fully connected layer with L2 regularization.
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(L2_LAMBDA)),

    # Apply dropout to prevent overfitting.
    Dropout(DROPOUT_RATE),

     # Output layer for binary classification using sigmoid activation.
    Dense(1, activation='sigmoid')
])

# Compile model using Adam optimizer and binary crossentropy loss.
import tensorflow as tf
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("CNN model built and compiled.")
model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configure early stopping to prevent overfitting.
early_stop = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, restore_best_weights=True)

# Configure learning rate reduction when validation loss stops improving.
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=REDUCE_LR_PATIENCE, verbose=1)

print("Callbacks ready for training.")

import time

# Define class weights to handle class imbalance.
class_weight = {0: 1.0, 1: 1.5}

start_time = time.time()

# Train the model using training and validation data.
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr],
     class_weight=class_weight
)
end_time = time.time()

elapsed_time = end_time - start_time
print("Model training complete.")
print(f"Training time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc

# Reset test generator before making predictions.
test_generator.reset()

# Predict probabilities for test data.
y_pred_probs = model.predict(test_generator)

# Convert probabilities into binary class predictions.
y_pred_classes = (y_pred_probs > 0.5).astype(int).reshape(-1)

# Get true labels from test generator.
y_true = test_generator.classes

# Confusion Matrix.
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("Confusion matrix displayed.")

# Classification Metrics.
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=list(test_generator.class_indices.keys())))

# AUC-ROC and ROC Curve.
auc_score = roc_auc_score(y_true, y_pred_probs)
fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
print(f"ROC curve plotted. AUC: {auc_score:.4f}")

# Training Curves.
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
print("Training curves plotted.")

# Sample Predictions.
import random
from tensorflow.keras.preprocessing import image

sample_indices = random.sample(range(len(test_generator.filenames)), 5)

plt.figure(figsize=(15,5))
for i, idx in enumerate(sample_indices):
    img_path = test_generator.filepaths[idx]
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred_prob = model.predict(img_array)[0][0]
    pred_label = "Pneumonia" if pred_prob > 0.5 else "Normal"
    true_label = test_generator.classes[idx]
    true_label_name = list(test_generator.class_indices.keys())[true_label]

    plt.subplot(1,5,i+1)
    plt.imshow(img)
    plt.title(f"True: {true_label_name}\nPred: {pred_label}")
    plt.axis('off')

plt.show()
print("Sample predictions displayed.")

# Comparison.
# ResNet50 (without top classifier).
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load pretrained ResNet50 model without top classification layer.
base_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Freeze early layers to retain pretrained features.
for layer in base_resnet.layers[:143]:
    layer.trainable = False

# Unfreeze deeper layers for fine-tuning.
for layer in base_resnet.layers[143:]:
    layer.trainable = True

x = GlobalAveragePooling2D()(base_resnet.output)
x = Dense(128, activation='relu')(x)
x = Dropout(DROPOUT_RATE)(x)
output = Dense(1, activation='sigmoid')(x)

resnet_model = Model(inputs=base_resnet.input, outputs=output)

resnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

resnet_history = resnet_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight
)

def evaluate_full_model(model, history, model_name):
    test_generator.reset()
    y_probs = model.predict(test_generator)
    y_preds = (y_probs > 0.5).astype(int).reshape(-1)
    y_true = test_generator.classes
    
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_true, y_preds))
    
    auc_score = roc_auc_score(y_true, y_probs)
    print(f"{model_name} AUC:", auc_score)
    
    return auc_score

auc_resnet = evaluate_full_model(resnet_model, resnet_history, "ResNet50")

# VGG16 (without top classifier).
from tensorflow.keras.applications import VGG16

# Load pretrained VGG16 model without top classification layer.
base_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Freeze early layers of VGG16.
for layer in base_vgg.layers[:15]:
    layer.trainable = False

# Unfreeze deeper layers for fine-tuning.
for layer in base_vgg.layers[15:]:
    layer.trainable = True

x = GlobalAveragePooling2D()(base_vgg.output)
x = Dense(128, activation='relu')(x)
x = Dropout(DROPOUT_RATE)(x)
output = Dense(1, activation='sigmoid')(x)

vgg_model = Model(inputs=base_vgg.input, outputs=output)

vgg_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

vgg_history = vgg_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight
)

auc_vgg = evaluate_full_model(vgg_model, vgg_history, "VGG16")

print("FINAL MODEL COMPARISON")
comparison_df = pd.DataFrame({
    "Model": ["Custom CNN", "ResNet50", "VGG16"],
    "AUC": [auc_score, auc_resnet, auc_vgg]
})

print(comparison_df)

print("FINAL VISUAL COMPARISON OF ALL MODELS")
# Reset test generator.
test_generator.reset()
y_true = test_generator.classes

# Custom CNN.
test_generator.reset()
y_probs_custom = model.predict(test_generator)
fpr_custom, tpr_custom, _ = roc_curve(y_true, y_probs_custom)

# ResNet50.
test_generator.reset()
y_probs_resnet = resnet_model.predict(test_generator)
fpr_resnet, tpr_resnet, _ = roc_curve(y_true, y_probs_resnet)

# VGG16.
test_generator.reset()
y_probs_vgg = vgg_model.predict(test_generator)
fpr_vgg, tpr_vgg, _ = roc_curve(y_true, y_probs_vgg)

plt.figure(figsize=(8,7))
plt.plot(fpr_custom, tpr_custom, label=f"Custom CNN (AUC={auc_score:.3f})")
plt.plot(fpr_resnet, tpr_resnet, label=f"ResNet50 (AUC={auc_resnet:.3f})")
plt.plot(fpr_vgg, tpr_vgg, label=f"VGG16 (AUC={auc_vgg:.3f})")
plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8,6))
plt.plot(history.history['val_accuracy'], label="Custom CNN")
plt.plot(resnet_history.history['val_accuracy'], label="ResNet50")
plt.plot(vgg_history.history['val_accuracy'], label="VGG16")
plt.title("Validation Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8,6))
plt.plot(history.history['val_loss'], label="Custom CNN")
plt.plot(resnet_history.history['val_loss'], label="ResNet50")
plt.plot(vgg_history.history['val_loss'], label="VGG16")
plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()


test_generator.reset()
pred_custom = (model.predict(test_generator) > 0.5).astype(int)

test_generator.reset()
pred_resnet = (resnet_model.predict(test_generator) > 0.5).astype(int)

test_generator.reset()
pred_vgg = (vgg_model.predict(test_generator) > 0.5).astype(int)

cm_custom = confusion_matrix(y_true, pred_custom)
cm_resnet = confusion_matrix(y_true, pred_resnet)
cm_vgg = confusion_matrix(y_true, pred_vgg)

fig, axes = plt.subplots(1,3, figsize=(18,5))

sns.heatmap(cm_custom, annot=True, fmt='d', cmap="Blues", ax=axes[0])
axes[0].set_title("Custom CNN")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(cm_resnet, annot=True, fmt='d', cmap="Greens", ax=axes[1])
axes[1].set_title("ResNet50")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

sns.heatmap(cm_vgg, annot=True, fmt='d', cmap="Oranges", ax=axes[2])
axes[2].set_title("VGG16")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")

plt.tight_layout()
plt.show()

comparison_df = pd.DataFrame({
    "Model": ["Custom CNN", "ResNet50", "VGG16"],
    "AUC": [auc_score, auc_resnet, auc_vgg],
    "Best Val Accuracy": [
        max(history.history['val_accuracy']),
        max(resnet_history.history['val_accuracy']),
        max(vgg_history.history['val_accuracy'])
    ]
})

print("\nFINAL MODEL COMPARISON TABLE:")
print(comparison_df)