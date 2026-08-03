import streamlit as st
import cv2
from ultralytics import YOLO
from utils import traducir_etiqueta # Asegúrate de tener tu utils.py en la misma carpeta
import torch
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# Configuración de la página
st.set_page_config(
    page_title="Industrial ZeroAccident - Tiempo Real",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Detección de EPP en TIEMPO REAL (WebRTC)")
st.markdown("Acércate a la cámara y prueba ponerte/quitarte el casco o chaleco.")

# --- CONFIGURACIÓN DEL MODELO YOLO ---
@st.cache_resource
def cargar_modelo_webrtc():
    model = YOLO("runs/detect/industrial_safety/ppe_yolov8n_run1-2/weights/best.pt")
    # Forzar dispositivo (preferiblemente GPU/MPS si está disponible para velocidad)
    if torch.backends.mps.is_available():
        model.to("mps")
    elif torch.cuda.is_available():
        model.to("cuda")
    else:
        model.to("cpu")
    return model

try:
    model_yolo = cargar_modelo_webrtc()
    nombres_clases = model_yolo.names
except Exception as e:
    st.error(f"Error al cargar el modelo YOLO: {e}")
    st.stop()

# Control de confianza en la barra lateral
confianza_slider = st.sidebar.slider("Umbral de Confianza", 0.1, 0.9, 0.5, 0.05)

# --- CLASE PROCESADORA DE VIDEO (EL CORAZÓN DE WEBRTC) ---
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        # Inicializamos variables que usaremos dentro del procesamiento
        self.confianza = 0.5
        self.modelo = model_yolo # Usamos el modelo cargado globalmente
        self.clases = nombres_clases

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Este método se llama automáticamente en cada cuadro (frame) de video
        
        # 1. Convertir el frame de video (formato AVFrame) a imagen OpenCV (numpy array)
        img = frame.to_ndarray(format="bgr24")

        # 2. Ejecutar la inferencia de YOLOv8 sobre la imagen
        # IMPORTANTE: Usamos stream=False para obtener resultados inmediatos en este frame
        # Ajusta imgsz según tu modelo (ej. 320, 416, 640)
        results = self.modelo(img, conf=self.confianza, imgsz=416, verbose=False)
        res = results[0]

        # 3. Dibujar las detecciones (bounding boxes y etiquetas) sobre la imagen
        # res.plot() devuelve la imagen anotada en formato BGR
        img_anotada = res.plot(conf=True, labels=True, boxes=True)

        # Opcional: Traducción de etiquetas en tiempo real (más complejo de inyectar aquí 
        # directamente sobre res.plot() de Ultralytics sin modificar la imagen, 
        # pero res.plot() usa los nombres originales del modelo. Para la demo, 
        # usaremos los nombres que ya tiene el modelo).

        # 4. Convertir la imagen anotada de vuelta a formato AVFrame para que WebRTC la muestre
        return av.VideoFrame.from_ndarray(img_anotada, format="bgr24")

# --- INTERFAZ DE STREAMLIT CON WEBRTC ---
# Configuramos el widget webrtc_streamer
webrtc_ctx = webrtc_streamer(
    key="yolo-epp-realtime",
    mode=WebRtcMode.SENDRECV, # Permite enviar video (cámara) y recibirlo (procesado)
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, # Configuración ICE básica para red
    video_processor_factory=YOLOVideoProcessor, # Indicamos nuestra clase procesadora
    media_stream_constraints={"video": True, "audio": False}, # Solo video
    async_processing=True # Procesamiento asíncrono para mayor fluidez
)

# --- ACTUALIZAR LA CONFIANZA EN TIEMPO REAL ---
# Si el stream está activo, actualizamos el valor de confianza dentro del objeto procesador
if webrtc_ctx.video_processor:
    # Esto accede a la instancia de YOLOVideoProcessor que está corriendo en el hilo de video
    webrtc_ctx.video_processor.confianza = confianza_slider

st.markdown("---")
st.info("Nota: La velocidad de detección depende de la potencia de tu CPU/GPU y de la resolución de la cámara. Si el video es lento, intenta reducir la resolución en la configuración del navegador.")

if not webrtc_ctx.state.playing:
    st.warning("👆 Haz clic en 'START' para activar la cámara y comenzar la detección en tiempo real.")