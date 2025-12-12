# c:\Users\rodri\ProyectosPython\agenteSAES_phi\main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
from utils_rag import ReglamentoRAG
from db_utils import obtener_datos_usuario, obtener_datos_profesor
from question_classifier import QuestionClassifier, DirectAnswerBuilder
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from cachetools import TTLCache
from threading import Lock, RLock
from typing import Dict, Any, Tuple, Optional
import re
import logging
import unicodedata
import time
import asyncio
import hashlib
import os
import uuid
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ============================================================================ 
# CONFIGURACIÓN Y CACHÉ
# ============================================================================ 
executor = ThreadPoolExecutor(max_workers=8)

# Caché con tiempo de vida (TTL)
cache_usuarios = TTLCache(maxsize=1000, ttl=300)   # 5 minutos
cache_respuestas = TTLCache(maxsize=500, ttl=600)  # 10 minutos

# Locks para acceso seguro a recursos compartidos
llm_lock = Lock()
cache_usuarios_lock = RLock()
cache_respuestas_lock = RLock()

# ============================================================================ 
# SISTEMA DE COLA DE MENSAJES
# ============================================================================ 
@dataclass
class QueueRequest:
    """Estructura para peticiones en la cola."""
    request_id: str
    pregunta: 'Pregunta'
    future: asyncio.Future
    timestamp: float

# Cola de mensajes para procesar peticiones secuencialmente
message_queue: asyncio.Queue = None
queue_stats = {
    "total_processed": 0,
    "total_errors": 0,
    "current_queue_size": 0,
    "processing": False,
}
queue_stats_lock = Lock()

# Inicialización de Modelos y RAG
try:
    model_path = os.environ.get(
        "LLM_MODEL_PATH",
        r"C:\Users\rodri\ProyectosPython\agenteSAES_phi\models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    )
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=4,
        n_batch=512,
        verbose=False,
    )
    rag = ReglamentoRAG(index_path="reglamentos_ipn.index", json_path="reglamentos_ipn.json")
    logging.info("Modelos LLM y RAG cargados correctamente.")
except Exception as e:
    logging.error(f"Error al cargar modelos: {e}", exc_info=True)
    llm = None
    rag = None

# ============================================================================ 
# ESQUEMAS
# ============================================================================ 
class Pregunta(BaseModel):
    query: str
    id_usuario: str
    tipo_usuario: str  # "alumno" o "profesor"
    razonamiento: int = 0  # 0 = usar clasificador (rapido), 1 = ir directo a LLM (razonamiento completo)


class CacheStats(BaseModel):
    cache_usuarios_size: int
    cache_respuestas_size: int


# ============================================================================ 
# PROMPT Y FUNCIONES AUXILIARES
# ============================================================================ 
PROMPT_SISTEMA_BASE = (
    "Eres un asistente académico del IPN (Instituto Politécnico Nacional de México). Usuario: **{tipo_usuario_upper}**.\n\n"
    "CONTEXTO: Estás respondiendo preguntas sobre educación, reglamentos académicos y trámites escolares del IPN. "
    "Todas las preguntas son en contexto educativo. Términos como 'ETS' se refieren a 'Evaluación a Título de Suficiencia' (un tipo de examen).\n\n"
    "REGLA FUNDAMENTAL: Solo puedes responder usando la información que aparece en los CONTEXTOS de abajo. "
    "NO uses tu conocimiento general. Si la respuesta NO está en los contextos, di: 'No tengo esa información en mi base de datos actual.'\n\n"
    "FORMATO DE HORARIOS: Los horarios están en formato 'Día HH:MM-HH:MM' donde HH:MM es hora de inicio y el segundo HH:MM es hora de fin. "
    "Ejemplo: 'Lunes 7:00-8:30' significa que la clase es el lunes de 7:00 AM a 8:30 AM.\n\n"
    "EJEMPLO DE CÓMO RESPONDER:\n"
"Pregunta: ¿Qué es un crédito?\n"
"Contexto: 'Crédito: A la unidad de reconocimiento académico que mide y cuantifica las actividades de aprendizaje contempladas en un plan de estudio.'\n"
"Respuesta CORRECTA: Un crédito es la unidad de reconocimiento académico que mide y cuantifica las actividades de aprendizaje contempladas en un plan de estudio, es universal y transferible entre programas académicos.\n\n"
"Pregunta: ¿A qué hora tengo clase de Matemáticas el lunes?\n"
"Contexto: 'Matemáticas (Gpo: 1A, Turno: Matutino) Horario: Lunes 7:00-8:30, Miércoles 10:00-11:30'\n"
"Respuesta CORRECTA: Tu clase de Matemáticas el lunes es de 7:00 a 8:30 AM.\n\n"
    "CONTEXTOS DISPONIBLES:\n\n"
    "=== DATOS DEL USUARIO ===\n"
    "{contexto_academico}\n\n"
    "=== REGLAMENTO IPN ===\n"
    "{contexto_rag}\n\n"
    "INSTRUCCIONES:\n"
    "1. Lee la pregunta del usuario\n"
    "2. Busca la respuesta SOLO en los contextos de arriba\n"
    "3. Si la encuentras: responde en 2-3 oraciones, un solo párrafo, sin listas\n"
    "4. Si NO la encuentras: di que no tienes esa información\n"
    "5. RESPONDE SIEMPRE EN ESPAÑOL\n\n"
)


def _construir_contexto_alumno(datos: Dict[str, Any]) -> str:
    """Construye el texto de contexto académico para el alumno."""
    if not datos or not datos.get("boleta"):
        return "No se pudo obtener información académica del alumno."
    contexto = [
        f"Boleta: {datos.get('boleta', 'N/A')}",
        f"Nombre: {datos.get('nombre', 'N/A')}",
        f"Carrera: {datos.get('carrera', 'N/A')}",
        f"Promedio general: {datos.get('promedio', 0.0):.2f}",
        f"Créditos disponibles: {datos.get('creditos_disponibles', 0)}",
        f"Estado académico: {datos.get('estado_academico', 'N/A')}",
        f"Situación en Kardex: {datos.get('situacion_kardex', 'N/A')}",
        f"Semestre Actual: {datos.get('semestre_actual', 'N/A')}",
        f"Reinscripción Activa: {'Sí' if datos.get('reinscripcion_activa') else 'No'} (Caduca: {datos.get('inscripcion_caduca', 'N/A')})",
        "\n--- Materias Inscritas ---\n" + (datos.get("materias_inscritas_texto", "Ninguna")),
        "\n--- Historial Académico (Aprobadas) ---\n" + (datos.get("materias_aprobadas_texto", "Sin materias aprobadas.")),
        "\n--- Historial Académico (Reprobadas) ---\n" + (datos.get("materias_reprobadas_texto", "Sin materias reprobadas.")),
        "\n--- Fechas Relevantes ---\n"
        + "\n".join(
            [f"- {k}: {v}" for k, v in datos.get("fechas_semestre", {}).items()
             if k in ["inicio_semestre", "fin_semestre", "registro_primer_parcial"]]
        ),
    ]
    return "\n".join(contexto)


def _construir_contexto_profesor(datos: Dict[str, Any]) -> str:
    """Construye el texto de contexto académico para el profesor."""
    if not datos or not datos.get("id_profesor"):
        return "No se pudo obtener información académica del profesor."
    contexto = [
        f"ID Profesor: {datos.get('id_profesor', 'N/A')}",
        f"Nombre: {datos.get('nombre', 'N/A')}",
        f"Grado: {datos.get('grado', 'N/A')}",
        f"Calificación promedio: {datos.get('calificacion_promedio', 0.0):.1f} ({datos.get('total_resenas', 0)} reseñas)",
        "\n--- Grupos Impartidos ---\n" + (datos.get("grupos_texto", "Sin grupos asignados.")),
        "\n--- Últimos Comentarios ---\n" + (datos.get("ultimos_comentarios", "Sin comentarios recientes.")),
        "\n--- Fechas Relevantes ---\n"
        + "\n".join(
            [f"- {k}: {v}" for k, v in datos.get("fechas_semestre", {}).items()
             if k in ["evalu_profe", "registro_primer_parcial", "fin_registro_primer_parcial"]]
        ),
    ]
    return "\n".join(contexto)




def _limpiar_respuesta(respuesta: str) -> str:
    """Post-procesa la respuesta para asegurar formato conciso y sin listas."""
    if not respuesta:
        return respuesta
    
    # Eliminar listas numeradas (1., 2., 3., etc.)
    respuesta = re.sub(r'^\s*\d+[\.\)]\s*', '', respuesta, flags=re.MULTILINE)
    
    # Eliminar viñetas (-, *, •)
    respuesta = re.sub(r'^\s*[-\*•]\s*', '', respuesta, flags=re.MULTILINE)
    
    # Convertir múltiples párrafos en uno solo (unir con espacio)
    respuesta = re.sub(r'\n\s*\n', ' ', respuesta)
    
    # Eliminar saltos de línea simples
    respuesta = re.sub(r'\n', ' ', respuesta)
    
    # Eliminar espacios múltiples
    respuesta = re.sub(r'\s+', ' ', respuesta)
    
    # Eliminar frases de relleno comunes
    frases_relleno = [
        r'Como asistente académico,?\s*',
        r'En resumen,?\s*',
        r'Para concluir,?\s*',
        r'Espero que esto ayude\.?\s*',
        r'Si tienes más preguntas\.?\s*',
    ]
    for frase in frases_relleno:
        respuesta = re.sub(frase, '', respuesta, flags=re.IGNORECASE)
    
    # Asegurar que termina con punto
    respuesta = respuesta.strip()
    if respuesta and respuesta[-1] not in '.!?]':
        respuesta += '.'
    
    return respuesta


def _validar_respuesta(respuesta: str) -> bool:
    """Valida que la respuesta cumpla con los criterios de calidad."""
    if not respuesta or len(respuesta) < 20:
        return False
    
    # Verificar que no contenga listas numeradas
    if re.search(r'^\s*\d+[\.\)]', respuesta, re.MULTILINE):
        return False
    
    # Verificar que no contenga viñetas
    if re.search(r'^\s*[-\*•]', respuesta, re.MULTILINE):
        return False
    
    # Verificar longitud razonable (50-500 caracteres)
    if len(respuesta) > 500:
        return False
    
    # Verificar que termina correctamente
    if respuesta[-1] not in '.!?]':
        return False
    
    return True

def _generar_respuesta_sync(prompt: str, texto_usuario: str) -> Tuple[str, float]:
    """Genera la respuesta usando Llama en un hilo."""
    if not llm:
        return "El modelo de lenguaje no está inicializado.", 0.0
    inicio = time.time()
    with llm_lock:
        try:
            output = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": texto_usuario},
                ],
                max_tokens=200,
                temperature=0.01,
                top_p=0.9,
                top_k=40,
                repeat_penalty=2.5,
            )
            respuesta = output["choices"][0]["message"]["content"].strip()
            tiempo_ms = round((time.time() - inicio) * 1000, 2)
            return respuesta, tiempo_ms
        except Exception as e:
            logging.error(f"Error en _generar_respuesta_sync: {e}")
            return f"Error en la generación de respuesta: {e}", 0.0


@lru_cache(maxsize=100)
def _buscar_contexto_cached(query: str, tipo_usuario: str, top_k: int = 3) -> str:
    """Busca contexto en el RAG con caché LRU e incorpora la expansión de query."""
    if not rag:
        return "El sistema RAG no está inicializado."
    expanded_query = query
    if tipo_usuario and tipo_usuario.lower() == "profesor":
        expanded_query = f"{query} docente enseñanza responsabilidades"
    elif tipo_usuario and tipo_usuario.lower() == "alumno":
        expanded_query = f"{query} estudiante requisitos académicos"
    logging.info(f"Query RAG expandida para {tipo_usuario}: {expanded_query}")
    contexto = rag.buscar_contexto(expanded_query, top_merge=top_k)
    return _dedup_sentences(contexto)

def _dedup_sentences(text: str) -> str:
    """Simplificación: limpia y elimina duplicados de fragmentos de texto."""
    if not text:
        return ""
    fragments = text.split("\n\n")
    unique_fragments = set()
    cleaned_fragments = []
    for fragment in fragments:
        cleaned = re.sub(r"\s+", " ", fragment).strip()
        norm = unicodedata.normalize("NFKD", cleaned).encode("ascii", "ignore").decode("ascii").lower()
        if cleaned and len(cleaned.split()) > 10 and norm not in unique_fragments:
            unique_fragments.add(norm)
            cleaned_fragments.append(cleaned)
    return "\n---\n".join(cleaned_fragments)


async def _process_single_request(pregunta: Pregunta) -> Dict[str, Any]:
    """Procesa una única petición (lógica original del endpoint)."""
    texto_usuario = pregunta.query
    id_usuario = pregunta.id_usuario
    tipo_usuario = pregunta.tipo_usuario.lower()
    razonamiento = pregunta.razonamiento
    logging.info(f"Procesando consulta de {tipo_usuario} {id_usuario}: {texto_usuario} (razonamiento={razonamiento})")

    # 1. Clasificacion (solo si razonamiento=0)
    if razonamiento == 1:
        tipo_pregunta = "complex"
        subtipo = None
        logging.info("Modo razonamiento activado: saltando clasificador, usando LLM directamente")
    else:
        tipo_pregunta, subtipo = QuestionClassifier.classify(texto_usuario)
        logging.info(f"Clasificacion: Tipo={tipo_pregunta}, Subtipo={subtipo}")

    # 2. Generar clave de caché
    cache_key = hashlib.sha256(f"{tipo_usuario}:{id_usuario}:{texto_usuario}".encode("utf-8")).hexdigest()
    with cache_respuestas_lock:
        if cache_key in cache_respuestas:
            logging.info("Respuesta obtenida de caché.")
            return {
                "response": cache_respuestas[cache_key],
                "tiempo_ms": 0,
                "tipo_respuesta": "direct" if tipo_pregunta == "direct" else "llm",
                "from_cache": True,
            }

    # 3. Obtener datos de usuario (caché)
    datos_usuario = _obtener_datos_usuario_cached(id_usuario, tipo_usuario)
    datos_encontrados = bool(datos_usuario and (datos_usuario.get("boleta") or datos_usuario.get("id_profesor")))

    # 4. Flujo de respuesta
    respuesta_final = None
    tiempo_ms = 0
    tipo_respuesta = "llm"

    # --- Lógica 1: Respuesta directa con fallback ---
    if tipo_pregunta == "direct" and subtipo:
        if subtipo.startswith("definicion_"):
            inicio = time.time()
            respuesta_final = DirectAnswerBuilder.build_answer(subtipo, datos_usuario or {})
            tiempo_ms = round((time.time() - inicio) * 1000, 2)
            tipo_respuesta = "direct"
            print("="*80)
            print(f"DEBUG - Respuesta Directa (Definición): {subtipo}")
            print(f"Longitud: {len(respuesta_final)} caracteres")
            print(f"Respuesta completa:\n{respuesta_final}")
            print("="*80)
        elif datos_encontrados:
            inicio = time.time()
            respuesta_directa = DirectAnswerBuilder.build_answer(subtipo, datos_usuario)
            tiempo_ms = round((time.time() - inicio) * 1000, 2)
            print("="*80)
            print(f"DEBUG - Respuesta Directa: {subtipo}")
            print(f"Longitud: {len(respuesta_directa)} caracteres")
            print(f"Respuesta completa:\n{respuesta_directa}")
            print("="*80)
            negaciones = [
                "No tienes grupos asignados",
                "Sin comentarios recientes",
                "No se pudo obtener información",
                "No cuentas con",
            ]
            if any(n in respuesta_directa for n in negaciones):
                logging.warning(f"Respuesta directa ({subtipo}) fue una negación/vacía. Cayendo a LLM.")
                tipo_pregunta = "complex"
            else:
                respuesta_final = respuesta_directa
                tipo_respuesta = "direct"
        else:
            logging.warning(f"Usuario {tipo_usuario} no encontrado para pregunta directa. Cayendo a LLM.")
            tipo_pregunta = "complex"

    # --- Lógica 2: LLM (complex o fallback) ---
    if tipo_pregunta == "complex" or respuesta_final is None:
        if tipo_usuario == "profesor":
            contexto_academico = _construir_contexto_profesor(datos_usuario or {})
        else:
            contexto_academico = _construir_contexto_alumno(datos_usuario or {})
        contexto_rag = _buscar_contexto_cached(texto_usuario, tipo_usuario)
        prompt_sistema = PROMPT_SISTEMA_BASE.format(
            tipo_usuario_upper=tipo_usuario.upper(),
            contexto_academico=contexto_academico,
            contexto_rag=contexto_rag,
        )
        full_prompt = f"{prompt_sistema}PREGUNTA: {texto_usuario}\n\nRESPUESTA (usa SOLO los contextos de arriba):"
        logging.info(f"Prompt generado (LLM): {len(full_prompt)} caracteres. Contexto RAG: {len(contexto_rag)} caracteres.")
        respuesta_llm, tiempo_ms = await asyncio.get_event_loop().run_in_executor(
            executor, _generar_respuesta_sync, full_prompt, texto_usuario
        )
        print("Full Prompt:")
        print(full_prompt)
        print("Respuesta LLM:")
        print(respuesta_llm)
        if not respuesta_llm or "Error en la generación de respuesta" in respuesta_llm:
            respuesta_final = "No pude generar una respuesta. Por favor, intenta reformular tu pregunta."
        else:
            respuesta_limpia = _limpiar_respuesta(respuesta_llm)
            if _validar_respuesta(respuesta_limpia):
                respuesta_final = respuesta_limpia
            else:
                respuesta_final = respuesta_limpia if len(respuesta_limpia) > 20 else respuesta_llm
            tipo_respuesta = "llm"

    # 5. Cachear respuesta (solo directas exitosas)
    if respuesta_final and tipo_respuesta == "direct":
        with cache_respuestas_lock:
            cache_respuestas[cache_key] = respuesta_final

    # DEBUG: Imprimir respuesta final antes de retornar
    print("="*80)
    print(f"DEBUG - RESPUESTA FINAL (antes de retornar)")
    print(f"Tipo: {tipo_respuesta}")
    print(f"Longitud: {len(respuesta_final) if respuesta_final else 0} caracteres")
    print(f"Respuesta completa:\n{respuesta_final}")
    print("="*80)

    return {
        "response": respuesta_final,
        "tiempo_ms": tiempo_ms,
        "tipo_respuesta": tipo_respuesta,
        "from_cache": False,
    }


async def queue_worker():
    """Worker que procesa peticiones de la cola secuencialmente."""
    logging.info("Queue worker iniciado")
    while True:
        request: QueueRequest = await message_queue.get()
        
        with queue_stats_lock:
            queue_stats["processing"] = True
            queue_stats["current_queue_size"] = message_queue.qsize()
        
        try:
            logging.info(f"Procesando request {request.request_id} (en cola por {time.time() - request.timestamp:.2f}s)")
            
            # Procesar la petición
            resultado = await _process_single_request(request.pregunta)
            
            # Establecer el resultado en el Future
            request.future.set_result(resultado)
            
            with queue_stats_lock:
                queue_stats["total_processed"] += 1
            
            logging.info(f"Request {request.request_id} completado exitosamente")
            
        except Exception as e:
            logging.error(f"Error procesando request {request.request_id}: {e}", exc_info=True)
            
            # Establecer la excepción en el Future
            request.future.set_exception(e)
            
            with queue_stats_lock:
                queue_stats["total_errors"] += 1
        
        finally:
            message_queue.task_done()
            with queue_stats_lock:
                queue_stats["processing"] = False
                queue_stats["current_queue_size"] = message_queue.qsize()


@lru_cache(maxsize=100)
def _obtener_datos_usuario_cached(id_usuario: str, tipo_usuario: str) -> Optional[Dict]:
    """Obtiene datos de usuario con caché TTL."""
    with cache_usuarios_lock:
        cache_key = f"{tipo_usuario}:{id_usuario}"
        if cache_key in cache_usuarios:
            logging.info(f"Datos de usuario ({tipo_usuario}) obtenidos de caché.")
            return cache_usuarios[cache_key]
        if tipo_usuario.lower() == "alumno":
            datos = obtener_datos_usuario(id_usuario)
        elif tipo_usuario.lower() == "profesor":
            datos = obtener_datos_profesor(id_usuario)
        else:
            datos = None
        if datos:
            cache_usuarios[cache_key] = datos
            return datos
        return None


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Inicializa la cola de mensajes y el worker al arrancar la aplicación."""
    global message_queue
    message_queue = asyncio.Queue()
    asyncio.create_task(queue_worker())
    logging.info("Sistema de cola de mensajes inicializado")


@app.on_event("shutdown")
async def shutdown_event():
    """Limpia recursos al cerrar la aplicación."""
    logging.info("Cerrando sistema de cola de mensajes")


@app.post("/generate/")
async def responder(pregunta: Pregunta):
    """Endpoint principal que agrega peticiones a la cola y espera la respuesta."""
    request_id = str(uuid.uuid4())
    texto_usuario = pregunta.query
    id_usuario = pregunta.id_usuario
    tipo_usuario = pregunta.tipo_usuario.lower()
    
    # Log detallado para debugging
    logging.info(f"=" * 80)
    logging.info(f"Nueva petición recibida: {request_id}")
    logging.info(f"Query: {texto_usuario[:100]}...")
    logging.info(f"Usuario: {id_usuario} (Tipo: {tipo_usuario})")
    logging.info(f"Razonamiento: {pregunta.razonamiento}")
    logging.info(f"=" * 80)
    
    # OPTIMIZACIÓN: Verificar caché ANTES de agregar a la cola
    cache_key = hashlib.sha256(f"{tipo_usuario}:{id_usuario}:{texto_usuario}".encode("utf-8")).hexdigest()
    with cache_respuestas_lock:
        if cache_key in cache_respuestas:
            logging.info(f"Respuesta obtenida de caché (sin cola): {request_id}")
            return {
                "response": cache_respuestas[cache_key],
                "tiempo_ms": 0,
                "tipo_respuesta": "cached",
                "from_cache": True,
                "request_id": request_id,
            }
    
    # Crear Future para la respuesta
    future = asyncio.Future()
    
    # Crear request y agregarlo a la cola
    queue_request = QueueRequest(
        request_id=request_id,
        pregunta=pregunta,
        future=future,
        timestamp=time.time()
    )
    
    await message_queue.put(queue_request)
    
    with queue_stats_lock:
        queue_stats["current_queue_size"] = message_queue.qsize()
    
    logging.info(f"Request {request_id} agregado a la cola (tamaño: {message_queue.qsize()})")
    
    # Esperar la respuesta con timeout
    try:
        resultado = await asyncio.wait_for(future, timeout=240.0)  # 4 minutos timeout
        resultado["request_id"] = request_id
        return resultado
    except asyncio.TimeoutError:
        logging.error(f"Timeout esperando respuesta para request {request_id}")
        return {
            "response": "La solicitud excedió el tiempo de espera. Por favor, intenta de nuevo.",
            "tiempo_ms": 0,
            "tipo_respuesta": "error",
            "from_cache": False,
            "request_id": request_id,
            "error": "timeout"
        }
    except Exception as e:
        logging.error(f"Error esperando respuesta para request {request_id}: {e}")
        return {
            "response": "Ocurrió un error procesando tu solicitud.",
            "tiempo_ms": 0,
            "tipo_respuesta": "error",
            "from_cache": False,
            "request_id": request_id,
            "error": str(e)
        }


# ============================================================================ 
# ENDPOINTS DE MONITOREO DE COLA
# ============================================================================ 

@app.get("/queue/status")
async def get_queue_status():
    """Devuelve el estado actual de la cola de mensajes."""
    with queue_stats_lock:
        return {
            "queue_size": message_queue.qsize() if message_queue else 0,
            "processing": queue_stats["processing"],
            "total_processed": queue_stats["total_processed"],
            "total_errors": queue_stats["total_errors"],
        }


@app.get("/queue/stats")
async def get_queue_stats():
    """Devuelve estadísticas detalladas de la cola."""
    with queue_stats_lock:
        return {
            "current_queue_size": queue_stats["current_queue_size"],
            "processing": queue_stats["processing"],
            "total_processed": queue_stats["total_processed"],
            "total_errors": queue_stats["total_errors"],
            "success_rate": (
                queue_stats["total_processed"] / 
                (queue_stats["total_processed"] + queue_stats["total_errors"]) * 100
                if (queue_stats["total_processed"] + queue_stats["total_errors"]) > 0
                else 0.0
            ),
        }


# ============================================================================ 
# ENDPOINTS DE CONTROL (OPCIONALES)
# ============================================================================ 

@app.get("/cache/stats", response_model=CacheStats)
async def get_cache_stats():
    """Devuelve estadísticas del caché."""
    return CacheStats(
        cache_usuarios_size=len(cache_usuarios),
        cache_respuestas_size=len(cache_respuestas),
    )


@app.post("/cache/clear")
async def clear_cache():
    """Limpia todos los cachés."""
    with cache_usuarios_lock:
        cache_usuarios.clear()
    with cache_respuestas_lock:
        cache_respuestas.clear()
    _obtener_datos_usuario_cached.cache_clear()
    _buscar_contexto_cached.cache_clear()
    logging.info("Todos los cachés limpiados.")
    return {"message": "Cachés limpiados correctamente."}