# === Step 1: Import Libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping

# === Step 2: Load and Process the Data ===
data = pd.read_csv('winequality-red.csv', delimiter=';')

# Add a new feature for total acidity
data['total_acidity'] = data['fixed acidity'] + data['volatile acidity']

# Define new quality categories
def quality_category(q):
    if 0 <= q <= 4:
        return 'Χαμηλής Ποιότητας'
    elif q == 5:
        return 'Κατώτερης Μέτριας Ποιότητας'
    elif q == 6:
        return 'Ανώτερης Μέτριας Ποιότητας'
    elif q >= 7:
        return 'Υψηλής Ποιότητας'

# Apply the new categorization
data['quality_category'] = data['quality'].apply(quality_category)

# Normalize the features
scaler = MinMaxScaler()
features = data.drop(columns=['quality', 'quality_category'])
normalized_features = scaler.fit_transform(features)

# Encode the quality categories as numbers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['quality_category'])

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(normalized_features, labels)

# Split data into training (70%), validation (15%), and test (15%) sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# === Step 3: Build the Neural Network ===
model = Sequential([
    Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001), input_dim=X_train.shape[1]),
    BatchNormalization(),
    Dropout(0.4),
    GaussianNoise(0.1),  # Add noise to improve generalization
    Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(4, activation='softmax')  # 4 classes: Χαμηλής, Κατώτερης Μέτριας, Ανώτερης Μέτριας, Υψηλής Ποιότητας
])

# Compile the model with a learning rate scheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.8,
    staircase=True
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# === Step 4: Train the Model ===
early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=200,
                    batch_size=64,
                    callbacks=[early_stopping])

# === Step 5: Evaluate the Model ===
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# === Step 6: Visualize the Training History ===
plt.figure(figsize=(12, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# === Step 7: Generate and Plot Confusion Matrix ===
y_pred_classes = np.argmax(model.predict(X_test), axis=1)

conf_matrix = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# === Step 8: Print Classification Report ===
report = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_)
print("Classification Report:")
print(report)

# === Step 9: Save the Model ===
model.save('wine_quality_model.h5')
print("Model saved as 'wine_quality_model.h5'")
