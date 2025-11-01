# 🎭 Detector de Emociones Faciales en Tiempo Real

**Autor:** Joseph Efren Godos Zapata  
**Fecha:** 31/10/2025

---

## 🎯 Descripción

Este proyecto implementa un sistema de reconocimiento de emociones humanas a partir de imágenes o video en vivo, utilizando una **Red Neuronal Convolucional (CNN)** entrenada con un dataset de rostros etiquetados y una **interfaz gráfica intuitiva** desarrollada en Python.

El sistema detecta rostros en tiempo real mediante la cámara web o en imágenes cargadas manualmente, y clasifica la emoción predominante entre siete categorías:

| Emoción | Etiqueta | Emoji |
|---------|----------|-------|
| Enojado | `angry` | 😠 |
| Disgusto | `disgust` | 🤢 |
| Miedo | `fear` | 😨 |
| Feliz | `happy` | 😊 |
| Triste | `sad` | 😢 |
| Sorpresa | `surprise` | 😲 |
| Neutral | `neutral` | 😐 |

---

## ⚙️ Tecnologías Utilizadas

### Frameworks y Librerías
- **TensorFlow / Keras** - Entrenamiento y predicción del modelo CNN
- **OpenCV** - Detección de rostros y procesamiento de video
- **Tkinter** - Interfaz gráfica de usuario (GUI)
- **PIL (Pillow)** - Manipulación de imágenes
- **Matplotlib** - Visualización de métricas de entrenamiento
- **Scikit-learn** - Evaluación del modelo (matriz de confusión, reporte)
- **NumPy** - Operaciones numéricas

### Dataset
[Human Face Emotions (Kaggle)](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions)

---

## 📁 Estructura del Proyecto

```
Proyecto_ML/
│
├── train_model.py              # Script para entrenar el modelo CNN
├── emociones.py                # Aplicación GUI para detección en tiempo real
├── modelo_emociones.h5         # Modelo entrenado (generado automáticamente)
├── README.md                   # Documentación del proyecto
│
└── Data/                       # Dataset organizado por emociones
    ├── angry/                  # Imágenes de enojo
    ├── disgust/                # Imágenes de disgusto
    ├── fear/                   # Imágenes de miedo
    ├── happy/                  # Imágenes de felicidad
    ├── sad/                    # Imágenes de tristeza
    ├── surprise/               # Imágenes de sorpresa
    └── neutral/                # Imágenes neutrales
```

---

## 🚀 Instalación

### Requisitos Previos
- Python 3.9 o superior
- Cámara web (para detección en tiempo real)

### Instalación de Dependencias

```bash
pip install tensorflow opencv-python pillow matplotlib scikit-learn numpy
```

O usando el siguiente comando:

```bash
pip install tensorflow==2.13.0 opencv-python==4.8.0.76 pillow matplotlib scikit-learn numpy
```

---

## ▶️ Instrucciones de Uso

### 1. Entrenar el Modelo

Antes de usar la aplicación, debes entrenar el modelo con el dataset:

```bash
cd C:\Users\zapata\Desktop\Proyecto_ML
python train_model.py
```

**Resultados esperados:**
- Se generará el archivo `modelo_emociones.h5`
- Se mostrará la matriz de confusión
- Se visualizarán gráficas de precisión y pérdida
- Precisión típica: **85%–90%** en validación

### 2. Ejecutar la Aplicación

```bash
python emociones.py
```

---

## 🎮 Funcionalidades de la Aplicación

### Botones de Control

| Botón | Función |
|-------|---------|
| ▶️ **INICIAR CÁMARA** | Activa la cámara web y detecta emociones en tiempo real |
| ⏹️ **DETENER** | Detiene la captura de video |
| 🖼️ **CARGAR IMAGEN** | Analiza una imagen estática desde archivo |
| 📁 **CARGAR MODELO** | Permite seleccionar otro modelo `.h5` entrenado |

### Panel de Análisis

- **Barras de progreso** para cada emoción con colores distintivos
- **Porcentajes en tiempo real** de cada emoción detectada
- **Indicador de estado** del modelo cargado

---

## 📊 Arquitectura del Modelo CNN

```
Capa                    Salida              Parámetros
================================================================
Conv2D                  (62, 62, 32)        896
MaxPooling2D            (31, 31, 32)        0
Conv2D                  (29, 29, 64)        18,496
MaxPooling2D            (14, 14, 64)        0
Conv2D                  (12, 12, 128)       73,856
MaxPooling2D            (6, 6, 128)         0
Flatten                 (4608)              0
Dense                   (128)               589,952
Dropout (0.5)           (128)               0
Dense (Softmax)         (7)                 903
================================================================
Total parámetros: 684,103
```

**Hiperparámetros:**
- Tamaño de imagen: 64x64 píxeles
- Batch size: 32
- Épocas: 10
- Optimizador: Adam (lr=0.001)
- Función de pérdida: Categorical Crossentropy

---

## 📈 Resultados Esperados

### Métricas de Desempeño
- ✅ Precisión en entrenamiento: ~90%
- ✅ Precisión en validación: ~85-90%
- ✅ Detección de rostros en tiempo real: <30ms por frame
- ✅ Predicción de emociones: <50ms por rostro

### Visualización
- Gráficas de precisión y pérdida durante el entrenamiento
- Matriz de confusión con el desempeño por clase
- Reporte de clasificación detallado

---

## 🔧 Posibles Mejoras Futuras

- [ ] Integrar modelos preentrenados (ResNet, EfficientNet, VGG16)
- [ ] Soporte para múltiples rostros simultáneos en pantalla
- [ ] Exportar resultados a PDF o CSV con estadísticas
- [ ] Versión web con Flask o Streamlit
- [ ] Optimización para dispositivos móviles (TensorFlow Lite)
- [ ] Agregar reconocimiento de emociones por audio
- [ ] Implementar seguimiento temporal de emociones
- [ ] Modo de calibración personalizada por usuario

---

## 🐛 Solución de Problemas

### Error: "No se pudo acceder a la cámara"
**Solución:** Verifica que ninguna otra aplicación esté usando la cámara (Zoom, Teams, etc.)

### Error: "Modelo no encontrado"
**Solución:** Ejecuta primero `train_model.py` para generar `modelo_emociones.h5`

### Error: Baja precisión en predicciones
**Solución:** 
- Asegúrate de tener buena iluminación
- La cámara debe capturar el rostro frontalmente
- Entrena el modelo con más épocas (aumenta `EPOCHS` en `train_model.py`)

### Error: "ModuleNotFoundError"
**Solución:** Instala las dependencias faltantes:
```bash
pip install [nombre_del_modulo]
```

---

## 📌 Notas Importantes

⚠️ **Orden de emociones:** El modelo asume que las emociones están en el orden:  
`['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']`

⚠️ **Requisitos:** Requiere conexión a cámara web para el modo en vivo

⚠️ **Compatibilidad:** Desarrollado y probado en Windows 10/11 con Python 3.9.2

⚠️ **Dataset:** Las imágenes deben estar organizadas en carpetas con los nombres exactos de las emociones

---

## 📚 Referencias

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [OpenCV Face Detection](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
- [Keras Sequential Model](https://keras.io/guides/sequential_model/)
- [Human Face Emotions Dataset](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions)

---

## 📄 Licencia

Este proyecto es de **uso educativo y académico**.  
Prohibida su comercialización sin autorización expresa del autor.

---

## 👤

**Joseph Efren Godos Zapata**  

---

## ⭐ Agradecimientos

Gracias a la comunidad de Kaggle por proporcionar el dataset, y a los desarrolladores de TensorFlow, OpenCV y Tkinter por sus herramientas de código abierto.

---

**© 2025 Joseph Efren Godos Zapata** | Todos los derechos reservados
