# 🛡️ Industrial ZeroAccident

> Sistema inteligente de supervisión, auditoría y cumplimiento de Equipos de Protección Personal (EPP) en entornos industriales mediante visión por computadora y modelos de lenguaje de gran escala (LLM).

---

## 📋 Descripción del Proyecto
**Industrial ZeroAccident** es una solución tecnológica diseñada para automatizar la prevención de riesgos laborales en sectores de manufactura y construcción. Ante los retos normativos (como la NOM-017-STPS) y las limitaciones de las auditorías manuales, este sistema procesa flujos de video para detectar de manera proactiva omisiones en el uso de EPP (cascos, gafas, chalecos), complementándose con un panel analítico interactivo y un asistente conversacional inteligente.

---

## ✨ Características Principales
* **👁️ Visión por Computadora (YOLOv8):** Detección en tiempo real de elementos de protección personal y clasificación de infracciones con soporte de aceleración por hardware (CUDA/MPS).
* **💬 Asistente Conversacional (LLM):** Módulo de auditoría basado en inteligencia artificial para realizar consultas analíticas en lenguaje natural sobre los registros de seguridad.
* **📊 Dashboard Interactivo:** Panel de control desarrollado en Streamlit para visualizar KPIs de cumplimiento por turnos y zonas de planta.
* **📄 Reportes Ejecutivos:** Generación automática de reportes en PDF con evidencias fotográficas recortadas (*bounding boxes*) y marcas de tiempo.

---

## 🛠️ Tecnologías y Librerías
El proyecto está desarrollado en **Python** utilizando las siguientes herramientas clave:
* [Streamlit](https://streamlit.io/) - Interfaz web interactiva.
* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Modelo de detección de objetos.
* [OpenCV](https://opencv.org/) - Procesamiento digital de video e imágenes.
* Modelos de Lenguaje (LLM) - Para el análisis conversacional de datos.

---

## ⚙️ Configuración e Instalación Local

Si deseas clonar y ejecutar este proyecto en tu computadora, sigue estos pasos:

1. **Clona el repositorio:**
   ```bash
   git clone [https://github.com/TU_USUARIO/iza-epp.git](https://github.com/TU_USUARIO/iza-epp.git)
   cd iza-epp



## 📁 Estructura del Repositorio

* **`app.py`**: Interfaz y lógica principal de la aplicación en Streamlit.
* **`utils.py`**: Funciones auxiliares (traducciones, procesamiento de video o métricas).
* **`Model/`**: Carpeta que almacena los pesos de los modelos personalizados entrenados.
* **`yolov8n.pt`**: Modelo base de YOLOv8 preentrenado.
* **`requirements.txt`**: Dependencias de Python necesarias para el proyecto.
* **`runtime.txt`**: Versión de Python especificada para el entorno de despliegue.
* **`.devcontainer/`**: Configuración para contenedores de desarrollo.
* **`assets/`**: Recursos visuales e imágenes de apoyo.
* **`historial/`**: Registros o archivos históricos del proyecto.




---

## 💻 Requisitos del Sistema

* **Python** (versión 3.10 o 3.11 recomendada).
* **Git** instalado para clonar el repositorio.
* Archivo de pesos del modelo (`best.pt` dentro de la carpeta `Model/` o en la raíz, según lo requiera tu código).

---
