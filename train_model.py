import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# === CONFIGURACIÓN GENERAL ===
DATA_DIR = r"C:\Users\zaval\Desktop\Proyecto_Ml\Data"
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10

# === CARGA DE DATOS ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# === DEFINICIÓN DEL MODELO CNN ===
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# === COMPILAR EL MODELO ===
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === ENTRENAMIENTO ===
print("\n🔹 Entrenando el modelo, por favor espera...\n")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    verbose=1
)

# === GUARDAR MODELO ===
MODEL_PATH = os.path.join(os.getcwd(), "modelo_emociones.h5")
model.save(MODEL_PATH)
print(f"\n✅ Modelo guardado correctamente en: {MODEL_PATH}")

# === EVALUACIÓN ===
print("\n📊 Evaluando el modelo...\n")
val_generator.reset()
Y_pred = model.predict(val_generator)
y_pred = np.argmax(Y_pred, axis=1)

print('🔹 Matriz de confusión:')
print(confusion_matrix(val_generator.classes, y_pred))

print('\n🔹 Reporte de clasificación:')
target_names = list(val_generator.class_indices.keys())
print(classification_report(val_generator.classes, y_pred, target_names=target_names))

# === GRAFICAR CURVAS DE ENTRENAMIENTO ===
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del modelo')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()

print("\n🎯 ENTRENAMIENTO COMPLETO — TODO OK ✅")
