import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import threading

class EmotionDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎭 Detector de Emociones en Tiempo Real")
        self.root.geometry("1200x700")
        self.root.configure(bg='#1e1e2e')
        
        # Variables
        self.model = None
        self.camera = None
        self.is_running = False
        self.current_frame = None
        self.emotion_labels = []
        
        # Colores para emociones
        self.emotion_colors = {
            'angry': '#ff4444',
            'disgust': '#aa44ff',
            'fear': '#ff8844',
            'happy': '#44ff44',
            'sad': '#4444ff',
            'surprise': '#ffff44',
            'neutral': '#888888'
        }
        
        self.setup_ui()
        self.load_model_safe()
        
    def setup_ui(self):
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#1e1e2e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Título
        title_label = tk.Label(
            main_frame,
            text="🎭 DETECTOR DE EMOCIONES",
            font=("Arial", 24, "bold"),
            bg='#1e1e2e',
            fg='#00ff88'
        )
        title_label.pack(pady=(0, 20))
        
        # Frame de contenido
        content_frame = tk.Frame(main_frame, bg='#1e1e2e')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame izquierdo (video)
        left_frame = tk.Frame(content_frame, bg='#2e2e3e', relief=tk.RAISED, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        video_title = tk.Label(
            left_frame,
            text="📹 Cámara en Vivo",
            font=("Arial", 14, "bold"),
            bg='#2e2e3e',
            fg='#ffffff'
        )
        video_title.pack(pady=10)
        
        self.video_label = tk.Label(left_frame, bg='#000000')
        self.video_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Frame derecho (estadísticas)
        right_frame = tk.Frame(content_frame, bg='#2e2e3e', relief=tk.RAISED, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_frame.config(width=350)
        
        stats_title = tk.Label(
            right_frame,
            text="📊 Análisis de Emociones",
            font=("Arial", 14, "bold"),
            bg='#2e2e3e',
            fg='#ffffff'
        )
        stats_title.pack(pady=10)
        
        # Frame de barras de emociones
        self.emotions_frame = tk.Frame(right_frame, bg='#2e2e3e')
        self.emotions_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        self.emotion_bars = {}
        self.emotion_percentages = {}
        
        # Estado del modelo
        self.status_label = tk.Label(
            right_frame,
            text="⚪ Modelo: No cargado",
            font=("Arial", 10),
            bg='#2e2e3e',
            fg='#ffaa00'
        )
        self.status_label.pack(pady=10)
        
        # Frame de controles
        control_frame = tk.Frame(main_frame, bg='#1e1e2e')
        control_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Botones
        btn_style = {
            'font': ("Arial", 12, "bold"),
            'relief': tk.RAISED,
            'bd': 3,
            'width': 15,
            'height': 2
        }
        
        self.btn_start = tk.Button(
            control_frame,
            text="▶️ INICIAR CÁMARA",
            command=self.start_camera,
            bg='#00ff88',
            fg='#000000',
            **btn_style
        )
        self.btn_start.pack(side=tk.LEFT, padx=10)
        
        self.btn_stop = tk.Button(
            control_frame,
            text="⏹️ DETENER",
            command=self.stop_camera,
            bg='#ff4444',
            fg='#ffffff',
            state=tk.DISABLED,
            **btn_style
        )
        self.btn_stop.pack(side=tk.LEFT, padx=10)
        
        self.btn_image = tk.Button(
            control_frame,
            text="🖼️ CARGAR IMAGEN",
            command=self.load_image,
            bg='#4488ff',
            fg='#ffffff',
            **btn_style
        )
        self.btn_image.pack(side=tk.LEFT, padx=10)
        
        self.btn_model = tk.Button(
            control_frame,
            text="📁 CARGAR MODELO",
            command=self.load_model_dialog,
            bg='#ff88ff',
            fg='#000000',
            **btn_style
        )
        self.btn_model.pack(side=tk.LEFT, padx=10)
        
    def load_model_safe(self):
        """Carga el modelo si existe en la ruta predeterminada"""
        default_path = os.path.join(os.getcwd(), "modelo_emociones.h5")
        if os.path.exists(default_path):
            try:
                self.model = load_model(default_path)
                self.emotion_labels = list(self.model.layers[-1].output_shape)
                # Etiquetas comunes de emociones
                self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                self.status_label.config(text="🟢 Modelo: Cargado", fg='#00ff88')
                self.create_emotion_bars()
                messagebox.showinfo("✅ Éxito", "Modelo cargado correctamente")
            except Exception as e:
                self.status_label.config(text="🔴 Modelo: Error al cargar", fg='#ff4444')
        else:
            self.status_label.config(text="⚪ Modelo: No encontrado", fg='#ffaa00')
    
    def load_model_dialog(self):
        """Carga el modelo desde un archivo seleccionado"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar modelo",
            filetypes=[("Archivos H5", "*.h5"), ("Todos los archivos", "*.*")]
        )
        
        if file_path:
            try:
                self.model = load_model(file_path)
                self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                self.status_label.config(text="🟢 Modelo: Cargado", fg='#00ff88')
                self.create_emotion_bars()
                messagebox.showinfo("✅ Éxito", "Modelo cargado correctamente")
            except Exception as e:
                messagebox.showerror("❌ Error", f"No se pudo cargar el modelo:\n{str(e)}")
                self.status_label.config(text="🔴 Modelo: Error", fg='#ff4444')
    
    def create_emotion_bars(self):
        """Crea las barras de progreso para cada emoción"""
        for widget in self.emotions_frame.winfo_children():
            widget.destroy()
        
        self.emotion_bars.clear()
        self.emotion_percentages.clear()
        
        for emotion in self.emotion_labels:
            frame = tk.Frame(self.emotions_frame, bg='#2e2e3e')
            frame.pack(fill=tk.X, pady=8)
            
            label = tk.Label(
                frame,
                text=emotion.upper(),
                font=("Arial", 10, "bold"),
                bg='#2e2e3e',
                fg='#ffffff',
                width=10,
                anchor='w'
            )
            label.pack(side=tk.LEFT, padx=(0, 10))
            
            bar_frame = tk.Frame(frame, bg='#1e1e2e', height=25, relief=tk.SUNKEN, bd=2)
            bar_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            bar = tk.Canvas(bar_frame, bg='#1e1e2e', height=21, highlightthickness=0)
            bar.pack(fill=tk.BOTH, expand=True)
            
            percentage = tk.Label(
                frame,
                text="0%",
                font=("Arial", 10, "bold"),
                bg='#2e2e3e',
                fg='#ffffff',
                width=6,
                anchor='e'
            )
            percentage.pack(side=tk.LEFT, padx=(10, 0))
            
            color = self.emotion_colors.get(emotion, '#888888')
            self.emotion_bars[emotion] = (bar, color)
            self.emotion_percentages[emotion] = percentage
    
    def update_emotion_bars(self, predictions):
        """Actualiza las barras con las predicciones"""
        if not self.emotion_bars:
            return
        
        for i, emotion in enumerate(self.emotion_labels):
            if i < len(predictions):
                value = predictions[i] * 100
                bar, color = self.emotion_bars[emotion]
                
                bar.delete("all")
                width = bar.winfo_width()
                fill_width = int((width - 4) * predictions[i])
                
                if fill_width > 0:
                    bar.create_rectangle(
                        2, 2, fill_width + 2, 19,
                        fill=color,
                        outline=''
                    )
                
                self.emotion_percentages[emotion].config(text=f"{value:.1f}%")
    
    def start_camera(self):
        """Inicia la captura de video"""
        if self.model is None:
            messagebox.showwarning("⚠️ Advertencia", "Por favor, carga un modelo primero")
            return
        
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            messagebox.showerror("❌ Error", "No se pudo acceder a la cámara")
            return
        
        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_image.config(state=tk.DISABLED)
        
        threading.Thread(target=self.process_video, daemon=True).start()
    
    def stop_camera(self):
        """Detiene la captura de video"""
        self.is_running = False
        if self.camera:
            self.camera.release()
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_image.config(state=tk.NORMAL)
    
    def process_video(self):
        """Procesa el video en tiempo real"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                face_roi = frame[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (64, 64))
                face_roi = face_roi.astype('float32') / 255.0
                face_roi = np.expand_dims(face_roi, axis=0)
                
                predictions = self.model.predict(face_roi, verbose=0)[0]
                self.update_emotion_bars(predictions)
                
                emotion_idx = np.argmax(predictions)
                emotion = self.emotion_labels[emotion_idx]
                confidence = predictions[emotion_idx] * 100
                
                text = f"{emotion.upper()} {confidence:.1f}%"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
    
    def load_image(self):
        """Carga y analiza una imagen"""
        if self.model is None:
            messagebox.showwarning("⚠️ Advertencia", "Por favor, carga un modelo primero")
            return
        
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png"), ("Todos los archivos", "*.*")]
        )
        
        if file_path:
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                
                image = cv2.imread(file_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) == 0:
                    messagebox.showwarning("⚠️ Advertencia", "No se detectaron rostros en la imagen")
                    return
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    face_roi = image[y:y+h, x:x+w]
                    face_roi = cv2.resize(face_roi, (64, 64))
                    face_roi = face_roi.astype('float32') / 255.0
                    face_roi = np.expand_dims(face_roi, axis=0)
                    
                    predictions = self.model.predict(face_roi, verbose=0)[0]
                    self.update_emotion_bars(predictions)
                    
                    emotion_idx = np.argmax(predictions)
                    emotion = self.emotion_labels[emotion_idx]
                    confidence = predictions[emotion_idx] * 100
                    
                    text = f"{emotion.upper()} {confidence:.1f}%"
                    cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                img = Image.fromarray(image)
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
            except Exception as e:
                messagebox.showerror("❌ Error", f"Error al procesar la imagen:\n{str(e)}")
    
    def on_closing(self):
        """Maneja el cierre de la aplicación"""
        self.stop_camera()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()