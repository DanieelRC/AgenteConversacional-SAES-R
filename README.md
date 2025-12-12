# Agente SAES Phi

Asistente académico inteligente para el Instituto Politécnico Nacional (IPN), diseñado para responder preguntas sobre reglamentos, trámites escolares y datos académicos de alumnos y profesores. Utiliza RAG (Retrieval-Augmented Generation) y modelos LLM locales.

## Requisitos Previos

- **Python 3.10** o superior.
- **RAM**: Mínimo 8GB (recomendado 16GB para ejecutar el modelo Llama cuantizado).
- **Espacio en disco**: ~6GB para modelos y dependencias.

## Instalación

Sigue estos pasos para configurar el entorno de desarrollo:

### 1. Clonar el repositorio y crear entorno virtual

```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
# Windows:
.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/Activate.ps1
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Descargar modelo de lenguaje (Spacy)

El proyecto utiliza un modelo de Spacy para procesamiento de texto en español:

```bash
python -m spacy download es_core_news_sm
```

### 4. Configurar Base de Datos (MySQL)

El sistema requiere una base de datos MySQL para la información de usuarios y horarios.

1.  Asegúrate de tener instalado MySQL Server.
2.  Importa el esquema y datos iniciales:
    ```bash
    mysql -u root -p < ISSI_SAES.sql
    ```
    *Nota: El script `ISSI_SAES.sql` crea la base de datos `SAES` y usuario/contraseña por defecto.*
3.  Verifica las credenciales de conexión en `db_utils.py` y ajústalas si tu configuración de MySQL es diferente:
    ```python
    db_pool = pooling.MySQLConnectionPool(
        # ...
        user="root",
        password="root",  # Cambia esto por tu contraseña
        database="SAES"
    )
    ```

### 5. Configurar el Modelo LLM

Este proyecto utiliza `llama-cpp-python` para ejecutar modelos Llama localmente.

1.  Descarga el modelo **Meta-Llama-3.1-8B-Instruct** en formato GGUF (versión cuantizada recomendada: `Q4_K_M`).
    *   Puedes buscarlo en HuggingFace (ej. `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`) 
    *   [https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf]
2.  Coloca el archivo `.gguf` en la carpeta `models/`:
    ```
    agenteSAES_phi/
    └── models/
        └── Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
    ```
3.  (Opcional) Si tu archivo tiene otro nombre o ruta, configura la variable de entorno antes de ejecutar:
    ```powershell
    set LLM_MODEL_PATH="C:\ruta\a\tu\modelo.gguf"
    ```

## Generación de la Base de Conocimiento (Pipeline)

Antes de iniciar el servidor, necesitas procesar los documentos PDF (reglamentos) para crear el índice vectorizado.
En caso de no tener los archivos JSON y el índice FAISS, ejecuta el pipeline completo:
1.  Coloca tus archivos PDF en la carpeta `reglamentos/`.
2.  Ejecuta el pipeline completo:

```bash
python ejecutar_pipeline.py
```

Esto generará:
- `reglamentos_ipn.json`: Texto extraído y limpio.
- `reglamentos_ipn.index`: Índice FAISS para búsquedas rápidas.

También puedes ejecutar pasos individuales:
- `python ejecutar_pipeline.py --paso 1`: Solo extrae texto a JSON.
- `python ejecutar_pipeline.py --paso 3`: Solo regenera embeddings (útil si cambias el modelo de embeddings).

## Ejecución del Servidor

Una vez generados los índices y configurado el modelo, inicia la API REST:

```bash
uvicorn main:app --reload
```

La API estará disponible en: `http://localhost:8000`

### Documentación Interactiva de la API

Visita `http://localhost:8000/docs` para probar los endpoints directamente desde el navegador.

## Estructura del Proyecto

- `main.py`: Servidor FastAPI y lógica principal del asistente.
- `pipeline_completa.py`: Lógica de procesamiento de documentos (ETL).
- `ejecutar_pipeline.py`: Script CLI para ejecutar el pipeline.
- `utils_rag.py`: Utilidades para búsqueda vectorial (RAG).
- `question_classifier.py`: Clasificador de intención de preguntas.
- `db_utils.py`: Conexión y consultas a base de datos de usuarios (simulada o real).
