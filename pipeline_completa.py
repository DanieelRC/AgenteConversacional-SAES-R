"""
1. Extrae texto de PDFs
2. Genera fragmentos y palabras clave
3. Limpia ambigüedades léxicas del JSON
4. Genera embeddings y crea índice FAISS
"""

import os
import json
import re
import fitz
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import faiss
import spacy
from unidecode import unidecode
import unicodedata

CARPETA_PDFS = "reglamentos"
ARCHIVO_JSON_SALIDA = "reglamentos_ipn.json"
ARCHIVO_INDEX_FAISS = "reglamentos_ipn.index"
EMBEDDING_MODEL = "all-mpnet-base-v2"
LONGITUD_MAX_FRAGMENTO = 800
NUM_KEYWORDS = 8

nlp = spacy.load("es_core_news_sm")

def es_texto_relevante(texto: str) -> bool:
    """Valida si el texto es relevante para el reglamento."""
    # Normalizar texto para comparación
    texto_norm = texto.lower().strip()
    
    # Si el texto es muy corto, rechazar
    if len(texto_norm.split()) < 10:
        return False
    
    # Patrones de ruido o texto irrelevante (reducidos y simplificados)
    ruido_patterns = [
        r'www\.',
        r'http[s]?://',
        r'@',
        r'(?i)derechos?\s+reservados?',
        r'(?i)prohibida\s+su\s+reproduccion',
        r'(?i)todos\s+los\s+derechos'
    ]
    
    # Si contiene patrones de ruido claros, rechazar
    if any(re.search(pattern, texto_norm) for pattern in ruido_patterns):
        return False
    
    # Debe contener palabras clave del reglamento (reducidas a las más importantes)
    palabras_reglamento = {
        'articulo', 'alumno', 'estudiante', 'escolar', 
        'evaluacion', 'materia', 'credito'
    }
    
    # Debe contener al menos 1 palabra clave del reglamento
    texto_simple = unidecode(texto_norm)
    matches = sum(1 for palabra in palabras_reglamento if palabra in texto_simple)
    if matches < 1:
        return False
    
    return True

def limpiar_texto_inicial(texto: str) -> str:
    """Limpieza inicial mejorada del texto extraído del PDF."""
    if not texto:
        return ""
        
    # Normalizar Unicode de forma segura
    try:
        texto = unicodedata.normalize('NFKC', texto)
    except Exception:
        texto = unidecode(texto)
    
    # Limpiar caracteres especiales
    texto = re.sub(r'[^\w\s\.,;:¿?¡!áéíóúÁÉÍÓÚñÑ-]', ' ', texto)
    
    # Unificar espacios y saltos de línea
    texto = re.sub(r'\s+', ' ', texto)
    
    # Limpiar puntuación
    texto = re.sub(r'\s*([.,;:!?])\s*', r'\1 ', texto)
    
    # Normalizar guiones y otros caracteres
    texto = re.sub(r'[-‐‑‒–—―]+', '-', texto)
    texto = re.sub(r'[""'']', '"', texto)
    
    # Remover URLs y correos
    texto = re.sub(r'https?://\S+', '', texto)
    texto = re.sub(r'\S+@\S+\.\S+', '', texto)
    
    # Remover números de página y similares
    texto = re.sub(r'(?i)p[aá]g(?:ina)?\.?\s*\d+(?:\s*de\s*\d+)?', '', texto)
    texto = re.sub(r'^\s*\d+\s*$', '', texto, flags=re.MULTILINE)
    
    return texto.strip()


def extraer_texto_pdf(ruta_pdf: str) -> str:
    """Extrae texto de PDF con mejor manejo de formato."""
    doc = fitz.open(ruta_pdf)
    texto_total = ""
    
    for pagina in doc:
        # Extraer bloques de texto manteniendo formato
        bloques = pagina.get_text("blocks")
        
        for bloque in bloques:
            texto_bloque = bloque[4]
            
            # Ignorar bloques muy cortos (probablemente ruido)
            if len(texto_bloque.strip()) < 5:
                continue
                
            # Ignorar bloques que parecen ser encabezados/pies de página
            if re.search(r'(?i)(página|pag\.?)\s*\d+', texto_bloque):
                continue
                
            # Ignorar bloques que son solo números
            if re.match(r'^\s*\d+\s*$', texto_bloque):
                continue
            
            texto_total += texto_bloque + "\n"
    
    doc.close()
    return limpiar_texto_inicial(texto_total)


def fragmentar_texto(texto: str, longitud_max: int = 800) -> list:
    """Divide el texto en fragmentos más inteligentemente."""
    if not texto.strip():
        return []
    
    # Dividir por artículos primero
    fragmentos = []
    articulos = re.split(r'(?i)(?:ARTÍCULO|Art(?:ículo|iculo)?\.?)\s+\d+\.?', texto)
    
    for articulo in articulos:
        if not articulo.strip():
            continue
        
        # Dividir artículos largos por puntos
        oraciones = re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])', articulo.strip())
        
        actual = ""
        for oracion in oraciones:
            oracion = oracion.strip()
            if not oracion:
                continue
                
            # Si la oración cabe en el fragmento actual
            if len(actual) + len(oracion) <= longitud_max:
                actual = (actual + " " + oracion).strip()
            else:
                # Guardar fragmento actual si tiene contenido
                if actual:
                    fragmentos.append(actual)
                actual = oracion
        
        # Agregar último fragmento si existe
        if actual:
            fragmentos.append(actual)
    
    # Filtrar fragmentos
    fragmentos = [f.strip() for f in fragmentos if len(f.strip().split()) >= 10]
    
    return fragmentos

def generar_keywords_backup(texto: str, n: int = 8) -> list:
    """Genera palabras clave alternativas usando spaCy."""
    doc = nlp(texto)
    palabras = [token.lemma_.lower() for token in doc if token.pos_ in ["NOUN", "VERB", "PROPN"] and len(token) > 3]
    frecuencia = {}
    for p in palabras:
        frecuencia[p] = frecuencia.get(p, 0) + 1
    top = sorted(frecuencia.items(), key=lambda x: x[1], reverse=True)
    return [t[0] for t in top[:n]]


def generar_palabras_clave(kw_model, texto: str, n: int = 8) -> list:
    """Genera palabras clave con KeyBERT y respaldo de spaCy."""
    try:
        keywords = kw_model.extract_keywords(texto, keyphrase_ngram_range=(1, 2), stop_words='spanish', top_n=n)
        palabras = [k for k, _ in keywords if len(k) > 3]
        if len(palabras) < 3:
            palabras = generar_keywords_backup(texto, n)
        return palabras
    except Exception:
        return generar_keywords_backup(texto, n)


def paso_1_generar_json(kw_model):
    """PASO 1: Extrae texto de PDFs y genera JSON inicial."""
    print("\n" + "="*70)
    print("PASO 1: GENERACIÓN DE JSON DESDE PDFs")
    print("="*70)
    
    pdfs = [f for f in os.listdir(CARPETA_PDFS) if f.lower().endswith(".pdf")]
    print(f"Archivos PDF encontrados: {len(pdfs)}")
    
    if len(pdfs) == 0:
        print(f"No se encontraron archivos PDF en '{CARPETA_PDFS}'")
        return []

    todos_fragmentos = []

    for pdf in tqdm(pdfs, desc="Procesando PDFs"):
        ruta_pdf = os.path.join(CARPETA_PDFS, pdf)
        texto = extraer_texto_pdf(ruta_pdf)
        fragmentos = fragmentar_texto(texto, LONGITUD_MAX_FRAGMENTO)

        for i, frag in enumerate(fragmentos):
            palabras_clave = generar_palabras_clave(kw_model, frag, NUM_KEYWORDS)
            todos_fragmentos.append({
                "documento": pdf,
                "fragmento_id": f"{pdf}_{i}",
                "texto": frag,
                "palabras_clave": palabras_clave
            })

    # Validación adicional antes de guardar
    todos_fragmentos = [
        fragmento for fragmento in todos_fragmentos 
        if es_texto_relevante(fragmento["texto"])
    ]
    
    with open(ARCHIVO_JSON_SALIDA, "w", encoding="utf-8") as f:
        json.dump(todos_fragmentos, f, ensure_ascii=False, indent=2)
    
    print(f"Guardados {len(todos_fragmentos)} fragmentos en '{ARCHIVO_JSON_SALIDA}'")
    return todos_fragmentos


def limpiar_texto_avanzado(texto):
    """
    Limpia el texto eliminando ambigüedades léxicas:
    - Palabras cortadas con guión al final de línea
    - Espacios en blanco innecesarios
    - Múltiples espacios consecutivos
    """
    if not texto:
        return texto
    
    # Eliminar palabras cortadas con guión seguido de espacio o salto de línea
    texto = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', texto)
    
    # Eliminar guiones al final de palabras seguidos de espacio
    texto = re.sub(r'(\w+)-\s+', r'\1', texto)
    
    # Normalizar múltiples espacios a uno solo
    texto = re.sub(r'\s+', ' ', texto)
    
    # Eliminar espacios al inicio y final
    texto = texto.strip()
    
    return texto


def limpiar_palabras_clave(palabras):
    """
    Limpia las palabras clave eliminando:
    - Guiones al final
    - Espacios en blanco adicionales
    """
    palabras_limpias = []
    for palabra in palabras:
        # Eliminar guiones al final de la palabra
        palabra_limpia = palabra.rstrip('- ')
        # Eliminar espacios al inicio y final
        palabra_limpia = palabra_limpia.strip()
        # Solo agregar si no está vacía
        if palabra_limpia:
            palabras_limpias.append(palabra_limpia)
    
    return palabras_limpias


def paso_2_limpiar_json(datos):
    """PASO 2: Limpia ambigüedades léxicas del JSON."""
    print("\n" + "="*70)
    print("PASO 2: LIMPIEZA DE AMBIGÜEDADES LÉXICAS")
    print("="*70)
    
    cambios_texto = 0
    cambios_palabras = 0
    
    # Limpiar cada fragmento
    for i, fragmento in enumerate(datos):
        # Limpiar el texto
        texto_original = fragmento.get('texto', '')
        texto_limpio = limpiar_texto_avanzado(texto_original)
        if texto_original != texto_limpio:
            cambios_texto += 1
        fragmento['texto'] = texto_limpio
        
        # Limpiar palabras clave
        palabras_originales = fragmento.get('palabras_clave', [])
        palabras_limpias = limpiar_palabras_clave(palabras_originales)
        if palabras_originales != palabras_limpias:
            cambios_palabras += 1
        fragmento['palabras_clave'] = palabras_limpias
        
        if (i + 1) % 1000 == 0:
            print(f"  Procesados {i + 1} fragmentos...")
    
    # Guardar el archivo limpio
    with open(ARCHIVO_JSON_SALIDA, 'w', encoding='utf-8') as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)

    print(f"JSON limpiado guardado en '{ARCHIVO_JSON_SALIDA}'")

    return datos


def paso_3_generar_embeddings(embed_model, datos):
    """PASO 3: Genera embeddings y crea índice FAISS."""
    print("\n" + "="*70)
    print("PASO 3: GENERACIÓN DE EMBEDDINGS E ÍNDICE FAISS")
    print("="*70)
    
    if not datos:
        print("Error: No hay datos para generar embeddings")
        return
    
    print("Generando embeddings con SentenceTransformer...")
    textos = [item["texto"] + " " + " ".join(item["palabras_clave"]) for item in datos]
    
    if not textos:
        print("Error: No hay textos para procesar")
        return
        
    try:
        embeddings = np.array(embed_model.encode(textos, convert_to_numpy=True, show_progress_bar=True))
        
        if embeddings.size == 0:
            print("Error: No se generaron embeddings")
            return
            
        print(f"Dimensión de embeddings: {embeddings.shape}")
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, ARCHIVO_INDEX_FAISS)
        
        print(f"Índice FAISS guardado en '{ARCHIVO_INDEX_FAISS}'")
        print(f"Total de vectores: {index.ntotal}")
        
    except Exception as e:
        print(f"Error al generar embeddings: {e}")
        raise


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def ejecutar_pipeline_completa():
    print(f"\nConfiguración:")
    print(f"- Carpeta de PDFs: {CARPETA_PDFS}")
    print(f"- Archivo JSON: {ARCHIVO_JSON_SALIDA}")
    print(f"- Archivo índice: {ARCHIVO_INDEX_FAISS}")
    print(f"- Modelo embedding: {EMBEDDING_MODEL}")
    print(f"- Longitud máx. fragmento: {LONGITUD_MAX_FRAGMENTO}")
    print(f"- Número de keywords: {NUM_KEYWORDS}")
    
    print("\nInicializando modelos de ML...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    kw_model = KeyBERT(EMBEDDING_MODEL)
    print("Modelos cargados correctamente")
    
    # PASO 1: Generar JSON desde PDFs
    datos = paso_1_generar_json(kw_model)
    
    if len(datos) == 0:
        print("\nNo se generaron fragmentos. Proceso terminado.")
        return
    
    # PASO 2: Limpiar ambigüedades léxicas
    datos = paso_2_limpiar_json(datos)
    
    # PASO 3: Generar embeddings e índice FAISS
    paso_3_generar_embeddings(embed_model, datos)
    

def main():
    """Función principal con manejo de errores."""
    try:
        ejecutar_pipeline_completa()
    except KeyboardInterrupt:
        print("\n\nProceso interrumpido por el usuario.")
    except Exception as e:
        print(f"\n\nError durante la ejecución: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()