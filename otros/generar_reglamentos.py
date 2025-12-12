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

CARPETA_PDFS = "reglamentos"
ARCHIVO_JSON_SALIDA = "reglamentos_ipn.json"
ARCHIVO_INDEX_FAISS = "reglamentos_ipn.index"
EMBEDDING_MODEL = "all-mpnet-base-v2"
LONGITUD_MAX_FRAGMENTO = 800
NUM_KEYWORDS = 8

nlp = spacy.load("es_core_news_sm")


def limpiar_texto(texto: str) -> str:
    """Limpieza profunda del texto."""
    texto = unidecode(texto)  # eliminar tildes y caracteres especiales
    texto = texto.replace("- ", "")  # eliminar guiones de salto de línea
    texto = re.sub(r"\s+", " ", texto)
    texto = re.sub(r"([.,;:!?])(?=[^\s])", r"\1 ", texto)
    texto = texto.strip()
    return texto


def extraer_texto_pdf(ruta_pdf: str) -> str:
    """Extrae texto de PDF."""
    doc = fitz.open(ruta_pdf)
    texto_total = ""
    for pagina in doc:
        texto_total += pagina.get_text("text")
    doc.close()
    return limpiar_texto(texto_total)


def fragmentar_texto(texto: str, longitud_max: int = 800) -> list:
    """Divide el texto en fragmentos."""
    oraciones = re.split(r'(?<=[.!?]) +', texto)
    fragmentos, actual = [], ""
    for oracion in oraciones:
        if len(actual) + len(oracion) <= longitud_max:
            actual += " " + oracion
        else:
            fragmentos.append(actual.strip())
            actual = oracion
    if actual:
        fragmentos.append(actual.strip())
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
            # Fallback
            palabras = generar_keywords_backup(texto, n)
        return palabras
    except Exception:
        return generar_keywords_backup(texto, n)


def main():
    print("Inicializando modelos...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    kw_model = KeyBERT(EMBEDDING_MODEL)

    pdfs = [f for f in os.listdir(CARPETA_PDFS) if f.lower().endswith(".pdf")]
    print(f"Se encontraron {len(pdfs)} archivos PDF en '{CARPETA_PDFS}'")

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

    with open(ARCHIVO_JSON_SALIDA, "w", encoding="utf-8") as f:
        json.dump(todos_fragmentos, f, ensure_ascii=False, indent=2)
    print(f"Guardados {len(todos_fragmentos)} fragmentos en {ARCHIVO_JSON_SALIDA}")

    print("Generando embeddings con SentenceTransformer...")
    textos = [item["texto"] + " " + " ".join(item["palabras_clave"]) for item in todos_fragmentos]
    embeddings = np.array(embed_model.encode(textos, convert_to_numpy=True, show_progress_bar=True))

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, ARCHIVO_INDEX_FAISS)

    print(f"Índice FAISS guardado en {ARCHIVO_INDEX_FAISS}")
    print("Proceso completado con éxito.")


if __name__ == "__main__":
    main()
