import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# --- DICCIONARIO DE TRADUCCIÓN EXACTO (YOLO .yaml -> Español Amigable) ---
TRADUCCION_EPP = {
    # Infracciones / Falta de EPP
    "no_ear-protection": "Falta de Protección Auditiva",
    "no_glasses": "Falta de Gafas de Protección",
    "hand_noglove": "Falta de Guantes",
    "head_nohelmet": "Falta de Casco de Seguridad",
    "face_nomask": "Falta de Mascarilla / Careta",
    "barefoots": "Pies Descalzos (Sin Calzado)",
    "sandals": "Uso de Sandalias (Calzado Inadecuado)",
    "shoes": "Calzado No Adecuado",

    # Elementos Correctos / EPP Adecuado
    "ear-protection": "Protección Auditiva (Correcto)",
    "harness": "Arnés de Seguridad",
    "boots": "Botas de Seguridad (Correcto)",
    "face_mask": "Mascarilla / Careta (Correcto)",
    "glasses": "Gafas de Protección (Correcto)",
    "hand_glove": "Guantes de Protección (Correcto)",
    "head_helmet": "Casco de Seguridad (Correcto)",
    "vest": "Chaleco Reflectante (Correcto)",
    "person": "Personal en Planta"
}

def traducir_etiqueta(clase_ingles):
    """Traduce y limpia la etiqueta técnica devuelta por YOLO al español."""
    clase_limpia = clase_ingles.strip().lower()
    return TRADUCCION_EPP.get(clase_limpia, clase_ingles.capitalize())

def generar_pdf_reporte(nombre_archivo, datos_tabla, resumen_dict):
    """Genera el reporte PDF de auditoría con marca de tiempo y diseño profesional."""
    carpeta_reportes = "reportes"
    if not os.path.exists(carpeta_reportes):
        os.makedirs(carpeta_reportes)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo_final = os.path.join(carpeta_reportes, f"reporte_auditoria_{timestamp}.pdf")
    
    doc = SimpleDocTemplate(nombre_archivo_final, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elementos = []
    
    styles = getSampleStyleSheet()
    estilo_titulo = ParagraphStyle(
        'TituloReporte',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=15
    )
    
    # Título del reporte
    elementos.append(Paragraph("Reporte de Auditoría EPP - ZeroAccident", estilo_titulo))
    elementos.append(Spacer(1, 10))
    
    # Sección de Resumen (KPIs en texto)
    resumen_data = [
        [Paragraph("<b>Cumplimiento General</b>", styles['Normal']), Paragraph(str(resumen_dict.get('cumplimiento')), styles['Normal'])],
        [Paragraph("<b>Total de Incidencias</b>", styles['Normal']), Paragraph(str(resumen_dict.get('total_incidencias')), styles['Normal'])],
        [Paragraph("<b>Tiempo Analizado</b>", styles['Normal']), Paragraph(str(resumen_dict.get('tiempo_analizado')), styles['Normal'])],
    ]
    t_resumen = Table(resumen_data, colWidths=[200, 300])
    t_resumen.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EDF2F7")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    elementos.append(t_resumen)
    elementos.append(Spacer(1, 20))
    
    elementos.append(Paragraph("<b>Registro Detallado de Testigos e Infracciones</b>", styles['Heading2']))
    elementos.append(Spacer(1, 10))
    
    # Construcción de la tabla con imágenes físicas
    tabla_pdf_data = [["Tiempo", "Infracción", "Confianza", "Evidencia Visual"]]
    
    for fila in datos_tabla:
        tiempo, infraccion, confianza, img_path = fila
        
        if img_path and os.path.exists(img_path):
            img_celda = Image(img_path, width=60, height=45)
        else:
            img_celda = Paragraph("Sin imagen", styles['Normal'])
            
        tabla_pdf_data.append([tiempo, infraccion, confianza, img_celda])
        
    t_detalles = Table(tabla_pdf_data, colWidths=[70, 180, 80, 120])
    t_detalles.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#F7FAFC")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor("#2D3748")),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 9),
    ]))
    
    elementos.append(t_detalles)
    doc.build(elementos)
    
    # Retornamos la ruta final para que app.py sepa exactamente dónde quedó guardado
    return nombre_archivo_final





import os
import pandas as pd
from datetime import datetime

HISTORIAL_DIR = "historial"
HISTORIAL_FILE = os.path.join(HISTORIAL_DIR, "historial_auditorias.csv")

def guardar_resultado_auditoria(nombre_video, cumplimiento, total_infracciones, duracion, causas_infracciones):
    """
    Crea la carpeta 'historial' si no existe y añade un nuevo registro 
    de auditoría incluyendo el detalle de las causas de infracción al CSV.
    """
    if not os.path.exists(HISTORIAL_DIR):
        os.makedirs(HISTORIAL_DIR)
        
    nuevo_registro = {
        "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Video": nombre_video,
        "Cumplimiento_Pct": cumplimiento,
        "Infracciones": total_infracciones,
        "Duracion_Seg": duracion,
        "Causas_Detalle": causas_infracciones  # <--- NUEVO CAMPO DE CAUSAS
    }
    
    if os.path.exists(HISTORIAL_FILE):
        df = pd.read_csv(HISTORIAL_FILE)
        df = pd.concat([df, pd.DataFrame([nuevo_registro])], ignore_index=True)
    else:
        df = pd.DataFrame([nuevo_registro])
        
    df.to_csv(HISTORIAL_FILE, index=False)