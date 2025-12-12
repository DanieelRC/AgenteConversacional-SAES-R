from sentence_transformers import SentenceTransformer
import json
import numpy as np
import faiss
import os
import re
import unicodedata
import spacy
from spacy.lang.es.stop_words import STOP_WORDS
from collections import defaultdict

# Stopwords: spaCy + algunas personalizadas (normalizadas sin acentos)
STOPWORDS_ES = set(STOP_WORDS) | {
    "segun", "sera", "son", "ser", "fue", "eran", "mas"
}

# Carga del modelo spaCy (deshabilita componentes innecesarios para velocidad)
NLP_ES = spacy.load("es_core_news_sm", disable=["parser", "ner", "textcat"])


def _normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", " ", s.lower()).strip()
    return s


def _is_noise(texto: str) -> bool:
    """Detecta fragmentos editoriales, directorios y ruido"""
    t = _normalize_text(texto)
    ruido_patterns = [
        r"gaceta politecnica",
        r"organo informativo", 
        r"directorio",
        r"queda estrictamente prohibida", 
        r"numero extraordinario",
        r"impreso en",
        r"talleres", 
        r"edicion:", 
        r"licitud",
        r"permiso de circulacion",
        r"coordinacion editorial",
        r"colaboradores",
        r"codigo de etica",
        r"principios y valores",
        r"\b\d+\s*de\s*\d+\b",
        r"www\.",
        r"http://",
        r"@"
    ]
    palabras = t.split()
    # Rechazar si muy corto o contiene patrones de ruido
    if len(palabras) < 12:
        return True
    
    # Detectar caracteres Unicode sospechosos de manera más segura
    if any(ord(c) > 127 for c in t):
        return True
        
    return any(re.search(pattern, t) for pattern in ruido_patterns)


def _lemmas_es(texto: str) -> list[str]:
    """
    Tokeniza y lematiza en español, removiendo stopwords, puntuación y tokens cortos.
    Devuelve lemas normalizados (minúsculas y sin acentos).
    """
    doc = NLP_ES(texto)
    toks = []
    for tok in doc:
        if not tok.is_alpha:
            continue
        lem = _normalize_text(tok.lemma_)
        if len(lem) <= 2 or lem in STOPWORDS_ES:
            continue
        toks.append(lem)
    return toks


def _lexical_tokens(pregunta: str) -> list[str]:
    """Mejora la tokenización para evitar ruido"""
    # Limpiar caracteres no ASCII de forma segura
    pregunta = ''.join(c for c in pregunta if ord(c) < 128)
    pregunta = unicodedata.normalize('NFKC', pregunta)
    
    # Continuar con la tokenización normal
    return _lemmas_es(pregunta)


class ReglamentoRAG:
    def __init__(self, json_path: str = "reglamentos_ipn.json", index_path: str = "reglamentos_ipn.index"):
        """
        Carga el reglamento fragmentado con palabras clave y el índice FAISS.
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"No se encontró el archivo JSON: {json_path}")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No se encontró el archivo FAISS: {index_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # Texto (pueden venir vacíos)
        self.textos = [item["texto"] for item in self.data]
        print(f"Reglamentos cargados con {len(self.textos)} fragmentos.")

        # Índice invertido de lemas para búsqueda léxica eficiente
        self.doc_lemmas: list[set[str]] = []
        self.inv_index: dict[str, set[int]] = defaultdict(set)
        for i, t in enumerate(self.textos):
            if _is_noise(t):
                self.doc_lemmas.append(set())
                continue
            lemset = set(_lemmas_es(t))
            self.doc_lemmas.append(lemset)
            for lem in lemset:
                self.inv_index[lem].add(i)
        print("Índice léxico (lemmas) construido.")

        # Cargar modelo de embeddings y el índice FAISS
        self.embedder = SentenceTransformer("all-mpnet-base-v2")
        self.index = faiss.read_index(index_path)
        print("Índice FAISS cargado correctamente.")

    def buscar_contexto(self, pregunta: str, k_faiss: int = 30, max_chars: int = 2000, top_merge: int = 5):
        """
        Recuperación híbrida: FAISS + léxico + validación de relevancia
        Prioriza fragmentos que contienen artículos específicos
        """
        if not pregunta or len(pregunta.strip()) == 0:
            return ""

        pregunta_lower = pregunta.lower()
        
        # Diccionario de expansiones por término clave
        expansion_map = {
            "irregular": "situacion escolar alumno regular irregular acreditar asignatura articulo 79",
            "regular": "situacion escolar regular acreditar asignatura promedio articulo 79",
            "suficiencia": "titulo suficiencia extraordinario ordinario evaluacion articulo 34 35",
            "ets": "evaluacion titulo suficiencia examen extraordinario ordinario articulo 34 35",
            "espa": "evaluacion saberes previamente adquiridos acreditar unidad aprendizaje",
            "dictamen": "dictamen situacion escolar irregular comision consejo tecnico",
            "dictaminado": "dictaminado alumno situacion irregular dictamen autorizado",
            "reinscripcion": "reinscripcion articulo 19 20 promedio creditos reinscribirse",
            "baja": "baja temporal definitiva causara articulo 49 57",
            "promedio": "promedio calificacion minimo articulo 41 seis ocho",
            "credito": "credito valor academico articulo 9",
            "evaluacion": "evaluacion ordinaria extraordinaria articulo 31 33 34",
            "materias aprobadas": "acreditar aprobar kardex materias asignatura",
            "horario": "materias inscritas grupo turno profesor",
            "kardex": "kardex historial academico calificaciones materias aprobadas",
            "tutor": "tutor academico orientacion asesor trayectoria escolar",
            "movilidad": "movilidad academica intercambio convenio institucion extranjera",
            "servicio social": "servicio social requisito titulacion horas comunidad",
            "titulacion": "titulacion egreso titulo profesional tesis examen",
        }

        
        pregunta_expandida = pregunta_lower
        for clave, expansion in expansion_map.items():
            if clave in pregunta_lower:
                pregunta_expandida += f" {expansion}"

        # Búsqueda semántica (FAISS)
        q_emb = np.array(self.embedder.encode([pregunta_expandida], convert_to_numpy=True))
        _, indices = self.index.search(q_emb, k_faiss)
        faiss_hits = [i for i in indices[0] if 0 <= i < len(self.textos)]

        # Búsqueda léxica
        toks = set(_lexical_tokens(pregunta))
        candidatos_lex = set()
        for tok in toks:
            candidatos_lex |= self.inv_index.get(tok, set())

        lex_scores = []
        for idx in candidatos_lex:
            if idx < 0 or idx >= len(self.textos):
                continue
            if _is_noise(self.textos[idx]):
                continue
            # Contar coincidencias de lemas
            matches = sum(1 for tok in toks if tok in self.doc_lemmas[idx])
            if matches > 0:
                # Bonificar presencia de "Artículo" al inicio
                texto_inicio = self.textos[idx][:100].lower()
                bonus = 3.0 if "articulo" in texto_inicio else 1.0
                lex_scores.append((idx, matches * bonus))

        # Combinar scores
        combined_scores: dict[int, float] = {}
        
        # Score FAISS (menor peso)
        for rank, idx in enumerate(faiss_hits):
            if _is_noise(self.textos[idx]):
                continue
            combined_scores[idx] = combined_scores.get(idx, 0.0) + (2.0 / (1 + rank * 0.3))

        # Score léxico (mayor peso)
        for idx, score in lex_scores:
            combined_scores[idx] = combined_scores.get(idx, 0.0) + (score * 1.5)

        if not combined_scores:
            candidatos = []
        else:
            # Ordenar por score descendente
            sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            candidatos = [self.textos[i] for i, _ in sorted_items[:top_merge]]

        # Deduplicar y ensamblar
        vistos = set()
        acc = []
        
        for texto in candidatos:
            if len(acc) >= top_merge:
                break
            
            texto_limpio = re.sub(r"\s+", " ", texto.strip())
            texto_norm = _normalize_text(texto_limpio)
            
            # Evitar duplicados y fragmentos muy cortos
            if texto_norm in vistos or len(texto_norm) < 50:
                continue
            
            vistos.add(texto_norm)
            acc.append(texto_limpio)

        # Unir con separador claro
        resultado = "\n\n".join(acc)[:max_chars]
        
        # Validar que no esté vacío
        if not resultado or len(resultado.strip()) < 20:
            return ""
        
        return resultado