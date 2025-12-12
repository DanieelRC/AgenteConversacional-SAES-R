import json
import re
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

ENTRADA_JSON = "reglamento_2.json"
SALIDA_JSON = "reglamento_fragmentado.json"
SALIDA_INDEX = "reglamento.index"
EMBEDDER_MODEL = "all-mpnet-base-v2"

def limpiar_texto(texto: str) -> str:
    """Limpia saltos, espacios, puntos extra, etc."""
    texto = re.sub(r"\s+", " ", texto)
    texto = texto.replace("•", "").replace("·", "")
    return texto.strip()

def fragmentar_texto(texto: str, max_len: int = 800) -> list:
    """Divide textos largos en fragmentos manejables."""
    texto = limpiar_texto(texto)
    if len(texto) <= max_len:
        return [texto]
    partes = re.split(r"(?<=[.?!])\s+", texto)
    fragmentos, buffer = [], ""
    for p in partes:
        if len(buffer) + len(p) < max_len:
            buffer += " " + p
        else:
            fragmentos.append(buffer.strip())
            buffer = p
    if buffer:
        fragmentos.append(buffer.strip())
    return fragmentos


print(f"Leyendo archivo: {ENTRADA_JSON}")
with open(ENTRADA_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

fragmentos_finales = []

if "titulos" in data:
    print("Estructura detectada: {'titulos' → 'capitulos' → 'articulos'}")
    titulos = data["titulos"]

    for t in tqdm(titulos, desc="Procesando títulos"):
        encabezado_titulo = t.get("encabezado_titulos", "")
        capitulos = t.get("capitulos", [])
        for c in capitulos:
            encabezado_cap = c.get("encabezado_capitulos", "")
            articulos = c.get("articulos", [])
            for art in articulos:
                num = art.get("numero_articulos", "Sin número")
                cont = art.get("contenido_articulo", "")
                if not cont.strip():
                    continue
                partes = fragmentar_texto(cont)
                for i, frag in enumerate(partes):
                    fragmentos_finales.append({
                        "titulo": f"{encabezado_titulo} - {encabezado_cap} - Artículo {num} (Sección {i+1})",
                        "contenido": frag
                    })

elif "orphan_text" in data:
    print("Estructura detectada: {'orphan_text'}")
    textos = data["orphan_text"]
    for i, txt in enumerate(tqdm(textos, desc="Procesando textos sueltos")):
        partes = fragmentar_texto(txt)
        for j, frag in enumerate(partes):
            fragmentos_finales.append({
                "titulo": f"Texto suelto {i+1}.{j+1}",
                "contenido": frag
            })
else:
    raise ValueError("No se encontraron claves esperadas ('titulos' o 'orphan_text').")

print(f"Total de fragmentos generados: {len(fragmentos_finales)}")

# GUARDAR JSON RESULTANTE
with open(SALIDA_JSON, "w", encoding="utf-8") as f:
    json.dump(fragmentos_finales, f, ensure_ascii=False, indent=2)
print(f"Archivo guardado: {SALIDA_JSON}")

# GENERAR ÍNDICE FAISS
print("Generando embeddings y base FAISS...")
model = SentenceTransformer(EMBEDDER_MODEL)
embeddings = model.encode(
    [f["contenido"] for f in fragmentos_finales],
    convert_to_numpy=True,
    show_progress_bar=True
)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings, dtype=np.float32))
faiss.write_index(index, SALIDA_INDEX)

print(f"Índice FAISS guardado como: {SALIDA_INDEX}")
print("Proceso completado con éxito.")
