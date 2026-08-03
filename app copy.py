import streamlit as st
import os
import tempfile
import time
import cv2
import pandas as pd
import torch
from ultralytics import YOLO
from utils import generar_pdf_reporte, traducir_etiqueta  # <-- Importamos ambas funciones de utils
from openai import OpenAI
from pydantic import BaseModel, Field

# --- CREAR CARPETA PARA EVIDENCIAS SI NO EXISTE ---
CARPETA_EVIDENCIAS = "evidencias_audit"
if not os.path.exists(CARPETA_EVIDENCIAS):
    os.makedirs(CARPETA_EVIDENCIAS)

# --- ESQUEMA FLEXIBLE PARA STRUCTURED OUTPUTS EN EL CHAT ---
class RespuestaConversacional(BaseModel):
    mensaje: str = Field(description="Respuesta natural y conversacional al usuario, manteniendo el rol de experto en seguridad industrial.")
    recomendacion_extra: str = Field(default="", description="Alguna recomendación adicional breve si el usuario lo pidió, de lo contrario dejar vacío.")

# Configuración de la página
st.set_page_config(
    page_title="Industrial ZeroAccident - Dashboard EPP",
    page_icon="",
    layout="wide"
)

# --- CARGAR EL MODELO YOLO CON ACELERACIÓN POR HARDWARE ---
@st.cache_resource
def cargar_modelo():
    model = YOLO("runs/detect/industrial_safety/ppe_yolov8n_run1-2/weights/best.pt")
    if torch.backends.mps.is_available():
        model.to("mps")
    elif torch.cuda.is_available():
        model.to("cuda")
    else:
        model.to("cpu")
    return model

try:
    model = cargar_modelo()
except Exception as e:
    st.sidebar.error(f"Error al cargar el modelo YOLO: {e}")

# --- DISEÑO DE LA BARRA LATERAL (SIDEBAR) ---
col_logo1, col_logo2, col_logo3 = st.sidebar.columns([1, 2, 1])
with col_logo2:
    st.image("assets/logo.png", width=420)

st.sidebar.markdown("<h1 style='text-align: center; font-size: 24px;'></h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.subheader("Carga de Video de Auditoría")

video_file = st.sidebar.file_uploader("Sube un video de la planta (.mp4, .mov)", type=["mp4", "mov", "avi"])
confianza_slider = st.sidebar.slider("Umbral de Confianza", 0.1, 0.9, 0.45, 0.05)
btn_procesar = st.sidebar.button("🚀 Analizar Video")




# --- CABECERA / BANNER ---
# Asegúrate de tener tu imagen rectangular (por ejemplo, 'banner_industrial.png') en la misma carpeta
st.image("banner_industrial.png", use_container_width=True)




# --- CUERPO PRINCIPAL (DASHBOARD) ---
st.title("Análisis de cumplimiento de EPP")
st.markdown("Monitoreo automatizado de seguridad industrial")

col1, col2, col3 = st.columns(3)
kpi_cumplimiento = col1.empty()
kpi_incidencias = col2.empty()
kpi_tiempo = col3.empty()

kpi_cumplimiento.metric(label="Cumplimiento del Video", value="-- %", delta="Pendiente")
kpi_incidencias.metric(label="Total de Incidencias", value="--", delta="0 infracciones")
kpi_tiempo.metric(label="Tiempo Analizado", value="00:00", delta="Duración")

st.markdown("---")

col_video, col_testigos = st.columns([1.5, 1])

with col_video:
    st.subheader("📹 Vista del Video")
    if video_file is not None:
        st.video(video_file)
    else:
        st.info("Sube un video en la barra lateral para visualizarlo aquí.")

with col_testigos:
    st.subheader("📋 Lista de Testigos / Incidencias")
    tabla_contenedor = st.empty()
    tabla_contenedor.info("Aquí aparecerá el registro tabular de eventos detectados tras el análisis.")

st.markdown("---")
st.subheader("📊 Temporalidad de Incidencias")
grafico_contenedor = st.empty()
grafico_contenedor.info("El gráfico de incidencias se generará tras procesar el video.")

st.subheader("📥 Reporte de Auditoría")
btn_descarga_contenedor = st.empty()
btn_descarga_contenedor.button("Descargar Reporte en PDF", disabled=True)


# --- LÓGICA DE PROCESAMIENTO ---
if btn_procesar:
    if video_file is None:
        st.error("Por favor, sube un video primero antes de presionar 'Analizar Video'.")
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        fps_real = cap.get(cv2.CAP_PROP_FPS)
        if fps_real <= 0:
            fps_real = 30.0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duracion_segundos = total_frames / fps_real if total_frames > 0 else 0
        
        min_tot = int(duracion_segundos // 60)
        seg_tot = int(duracion_segundos % 60)
        tiempo_str = f"{min_tot:02d}:{seg_tot:02d}"
        cap.release()

        barra_progreso = st.progress(0)
        texto_estado = st.empty()

        inicio_tiempo = time.time()

        results = model.predict(source=video_path, conf=confianza_slider, imgsz=416, save=False, stream=True)
        INTERVALO_FRAMES = 60  

        datos_tabla = []
        total_infracciones = 0
        frames_con_gente = 0
        frames_con_infraccion = 0
        
        infracciones_detectadas_crudo = []

        for numero_frame, r in enumerate(results):
            if total_frames > 0 and numero_frame % 30 == 0:  
                porcentaje = min(int((numero_frame / total_frames) * 100), 100)
                barra_progreso.progress(porcentaje)
                texto_estado.text(f"Analizando frame {numero_frame} de {total_frames} ({porcentaje}%) ...")

            if numero_frame % INTERVALO_FRAMES != 0:
                continue
            
            boxes = r.boxes
            if len(boxes) > 0:
                hay_infraccion_en_frame = False
                for box in boxes:
                    clase_id = int(box.cls[0])
                    nombre_clase_bruta = r.names[clase_id]
                    confianza = float(box.conf[0])

                    if "person" in nombre_clase_bruta.lower():
                        frames_con_gente += 1

                    if "no" in nombre_clase_bruta.lower():
                        hay_infraccion_en_frame = True
                        total_infracciones += 1
                        
                        seg_totales_frame = numero_frame / fps_real
                        m_f = int(seg_totales_frame // 60)
                        s_f = int(seg_totales_frame % 60)
                        tiempo_formato = f"{m_f:02d}:{s_f:02d}"

                        # Traducimos la clase usando la función importada de utils
                        clase_espanol = traducir_etiqueta(nombre_clase_bruta)

                        infracciones_detectadas_crudo.append({
                            'tiempo': tiempo_formato,
                            'clase': clase_espanol,
                            'conf': f"{confianza:.2f}",
                            'r_obj': r
                        })

                if hay_infraccion_en_frame:
                    frames_con_infraccion += 1

        contador_evidencias = 0
        for item in infracciones_detectadas_crudo:
            contador_evidencias += 1
            path_imagen_evidencia = None
            try:
                img_anotada = item['r_obj'].plot()
                nombre_archivo = f"evidencia_{contador_evidencias}.jpg"
                path_imagen_evidencia = os.path.join(CARPETA_EVIDENCIAS, nombre_archivo)
                cv2.imwrite(path_imagen_evidencia, img_anotada)
            except Exception as img_err:
                print(f"Error guardando evidencia: {img_err}")

            datos_tabla.append([item['tiempo'], item['clase'], item['conf'], path_imagen_evidencia])

        fin_tiempo = time.time()
        tiempo_proceso_segundos = fin_tiempo - inicio_tiempo

        barra_progreso.progress(100)
        texto_estado.success(f"¡Análisis completado en {tiempo_proceso_segundos:.2f} segundos!")

        if (frames_con_infraccion + frames_con_gente) > 0:
            porcentaje_cumplimiento = max(0, 100 - int((frames_con_infraccion / max(1, (frames_con_infraccion + frames_con_gente))) * 100))
        else:
            porcentaje_cumplimiento = 100  

        # --- GUARDAR EN SESSION_STATE PARA PERSISTENCIA ---
        st.session_state["porcentaje_cumplimiento"] = porcentaje_cumplimiento
        st.session_state["total_infracciones"] = total_infracciones
        st.session_state["tiempo_str"] = tiempo_str
        st.session_state["datos_tabla"] = datos_tabla
        st.session_state["video_procesado"] = True


# --- RENDERIZADO PERSISTENTE ---
if st.session_state.get("video_procesado", False):
    p_cump = st.session_state["porcentaje_cumplimiento"]
    t_infr = st.session_state["total_infracciones"]
    t_str = st.session_state["tiempo_str"]
    d_tabla = st.session_state["datos_tabla"]

    kpi_cumplimiento.metric(label="Cumplimiento del Video", value=f"{p_cump}%", delta="Nivel óptimo" if p_cump > 80 else "Requiere atención")
    kpi_incidencias.metric(label="Total de Incidencias", value=str(t_infr), delta="Infracciones EPP")
    kpi_tiempo.metric(label="Tiempo Analizado", value=t_str, delta="Duración real")

    if len(d_tabla) > 0:
        df_mostrar = pd.DataFrame(d_tabla, columns=["Tiempo", "Infracción", "Confianza", "Evidencia"])
        tabla_contenedor.dataframe(df_mostrar[["Tiempo", "Infracción", "Confianza"]], use_container_width=True)
        
        conteo_tiempo = df_mostrar.groupby("Tiempo").size().reset_index(name="Cantidad")
        grafico_contenedor.bar_chart(conteo_tiempo.set_index("Tiempo"))
    else:
        tabla_contenedor.success("¡Excelente! No se detectaron infracciones de EPP en los intervalos analizados.")
        grafico_contenedor.info("Sin datos suficientes para graficar.")

    resumen_dict = {
        'cumplimiento': f"{p_cump}%",
        'total_incidencias': t_infr,
        'tiempo_analizado': t_str
    }
    
    # Llamamos a la función modularizada en utils que maneja la carpeta 'reportes' y la fecha
    ruta_pdf_generado = generar_pdf_reporte("reporte_auditoria_epp.pdf", d_tabla, resumen_dict)

    with open(ruta_pdf_generado, "rb") as pdf_file:
        PDFbyte = pdf_file.read()

    btn_descarga_contenedor.download_button(
        label="📥 Descargar Reporte en PDF",
        data=PDFbyte,
        file_name=os.path.basename(ruta_pdf_generado),
        mime="application/pdf"
    )
    

    # ==========================================
    # --- ASISTENTE INTELIGENTE CON STRUCTURED OUTPUTS FLEXIBLE ---
    # ==========================================
    
    from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
    load_dotenv()
    
    
    
    st.markdown("---")
    st.subheader("💬 Asistente Experto en Seguridad (Auditoría actual)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    detalle_infracciones_texto = ""
    if len(d_tabla) > 0:
        for idx, row in enumerate(d_tabla, 1):
            detalle_infracciones_texto += f"{idx}. Tiempo: {row[0]} - Infracción detectada: {row[1]} (Confianza: {row[2]})\n"
    else:
        detalle_infracciones_texto = "No se registraron infracciones en este video."

    if prompt := st.chat_input("Ej: ¿Qué medidas correctivas debemos tomar según este cumplimiento?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analizando con el experto..."):
                
                system_prompt_estricto = f"""
                Eres un Auditor Senior y Experto en Seguridad Industrial y Equipos de Protección Personal (EPP). 
                Tienes acceso total a los resultados de la auditoría actual:
                - Porcentaje de cumplimiento: {p_cump}%
                - Total de infracciones detectadas: {t_infr}
                - Duración analizada: {t_str}
                
                DETALLE EXACTO DE LAS INCIDENCIAS ENCONTRADAS:
                {detalle_infracciones_texto}

                REGLAS DE COMPORTAMIENTO:
                1. Responde de forma natural y conversacional a lo que el usuario pregunte o salude.
                2. Basate estrictamente en los datos de la auditoría actual cuando te pidan explicaciones de fallas o recomendaciones.
                3. NUNCA respondas sobre temas completamente ajenos a la seguridad industrial o EPP.
                """
                
                try:
                    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    
                    completion = client.beta.chat.completions.parse(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt_estricto},
                            *st.session_state.messages
                        ],
                        response_format=RespuestaConversacional,
                        temperature=0.3,
                        max_tokens=1500
                    )
                    
                    resultado = completion.choices[0].message.parsed
                    
                    respuesta_final = resultado.mensaje
                    if resultado.recomendacion_extra:
                        respuesta_final += f"\n\n**Recomendación adicional:** {resultado.recomendacion_extra}"

                    st.markdown(respuesta_final)
                    st.session_state.messages.append({"role": "assistant", "content": respuesta_final})
                    
                except Exception as e:
                    st.error(f"Error al conectar con la API o procesar el Structured Output: {e}")
else:
    st.info("ℹ️ Sube y procesa un video en la barra lateral para habilitar al asistente experto con los resultados de la auditoría.")