# 🎭 Detector de Emociones Faciales en Tiempo Real

**Autor**: Joseph Efren Godos Zapata  
**Fecha**: 31/10/2025  

---

## 🎯 Descripción

Este proyecto implementa un sistema de reconocimiento de emociones humanas a partir de imágenes o video en vivo, utilizando una **Red Neuronal Convolucional (CNN)** entrenada con un dataset de rostros etiquetados y una **interfaz gráfica intuitiva** desarrollada en Python.

El sistema detecta rostros en tiempo real mediante la cámara web o en imágenes cargadas manualmente, y clasifica la emoción predominante entre siete categorías:
- 😠 Enojado (`angry`)
- 🤢 Disgusto (`disgust`)
- 😨 Miedo (`fear`)
- 😊 Feliz (`happy`)
- 😢 Triste (`sad`)
- 😲 Sorpresa (`surprise`)
- 😐 Neutral (`neutral`)

---

## ⚙️ Tecnologías Utilizadas

- **TensorFlow / Keras**: Entrenamiento y predicción del modelo CNN.
- **OpenCV**: Detección de rostros y procesamiento de video.
- **Tkinter**: Interfaz gráfica de usuario (GUI).
- **PIL (Pillow)**: Manipulación de imágenes.
- **Matplotlib**: Visualización de métricas de entrenamiento.
- **Scikit-learn**: Evaluación del modelo (matriz de confusión, reporte de clasificación).
- **NumPy, threading, os**: Soporte general.

Dataset utilizado: [Human Face Emotions (Kaggle)](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions)

---

## 📁 Estructura del Proyecto
Proyecto_ML/
├── train_model.py # Script para entrenar el modelo CNN
├── emociones.py # Aplicación GUI para detección en tiempo real
├── modelo_emociones.h5 # Modelo entrenado (generado tras ejecutar train_model.py)
├── Data/ # Carpeta con subcarpetas por emoción (dataset)
│ ├── angry/
│ ├── disgust/
│ ├── fear/
│ ├── happy/
│ ├── sad/
│ ├── surprise/
│ └── neutral/
└── README.md

---

## ▶️ Instrucciones de Uso

### Entrenar el modelo y ejecutar
```bash
python train_model.py
python emociones.py
Funcionalidades:
▶️ INICIAR CÁMARA: Detecta emociones en tiempo real.
⏹️ DETENER: Detiene la captura.
🖼️ CARGAR IMAGEN: Analiza una imagen estática.
📁 CARGAR MODELO: Permite seleccionar otro modelo .h5.
📊 Resultados Esperados
Precisión típica: 85%–90% en validación.
Interfaz visual con barras de progreso por emoción y porcentajes en tiempo real.
Detección robusta de rostros con Haar Cascade.
🔧 Posibles Mejoras Futuras
Integrar modelos preentrenados (ResNet, EfficientNet).
Soporte para múltiples rostros simultáneos.
Exportar resultados a PDF o CSV.
Versión web con Flask o Streamlit.
Optimización para dispositivos móviles (TensorFlow Lite).
📌 Notas
El modelo asume que las emociones están en el orden:
['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'].
Requiere conexión a cámara web para el modo en vivo.
Desarrollado y probado en Windows 10/11 con Python 3.9+.
📎 Licencia
Este proyecto es de uso educativo.
© 2025 Joseph Efren Godos Zapata

