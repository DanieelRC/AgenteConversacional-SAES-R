"""
Uso:
    python ejecutar_pipeline.py                  # Ejecuta todos los pasos
    python ejecutar_pipeline.py --paso 1         # Solo genera JSON
    python ejecutar_pipeline.py --paso 2         # Solo limpia JSON existente
    python ejecutar_pipeline.py --paso 3         # Solo genera embeddings
    python ejecutar_pipeline.py --desde 2        # Ejecuta desde el paso 2
"""

import argparse
import sys
from pipeline_completa import (
    paso_1_generar_json,
    paso_2_limpiar_json,
    paso_3_generar_embeddings,
    ARCHIVO_JSON_SALIDA,
    EMBEDDING_MODEL,
    SentenceTransformer,
    KeyBERT,
    json
)


def cargar_json_existente():
    """Carga el archivo JSON si existe."""
    try:
        with open(ARCHIVO_JSON_SALIDA, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encuentra el archivo '{ARCHIVO_JSON_SALIDA}'")
        print("Primero debes ejecutar el paso 1 para generar el JSON.")
        return None
    except json.JSONDecodeError:
        print(f"Error: El archivo '{ARCHIVO_JSON_SALIDA}' no es un JSON válido.")
        return None


def ejecutar_paso(paso_num, kw_model=None, embed_model=None):
    """Ejecuta un paso específico de la pipeline."""
    
    if paso_num == 1:
        print("\nEjecutando PASO 1: Generación de JSON desde PDFs")
        if kw_model is None:
            print("Cargando modelo KeyBERT...")
            kw_model = KeyBERT(EMBEDDING_MODEL)
        return paso_1_generar_json(kw_model)
    
    elif paso_num == 2:
        print("\nEjecutando PASO 2: Limpieza de ambigüedades léxicas")
        datos = cargar_json_existente()
        if datos is None:
            return None
        return paso_2_limpiar_json(datos)
    
    elif paso_num == 3:
        print("\nEjecutando PASO 3: Generación de embeddings e índice FAISS")
        datos = cargar_json_existente()
        if datos is None:
            return None
        if embed_model is None:
            print("Cargando modelo SentenceTransformer...")
            embed_model = SentenceTransformer(EMBEDDING_MODEL)
        paso_3_generar_embeddings(embed_model, datos)
        return datos
    
    else:
        print(f"Paso {paso_num} no válido. Los pasos válidos son 1, 2 o 3.")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Ejecutor modular de pipeline de procesamiento de reglamentos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python ejecutar_pipeline.py                    # Ejecuta toda la pipeline
  python ejecutar_pipeline.py --paso 1           # Solo extrae PDFs y genera JSON
  python ejecutar_pipeline.py --paso 2           # Solo limpia el JSON existente
  python ejecutar_pipeline.py --paso 3           # Solo genera embeddings del JSON existente
  python ejecutar_pipeline.py --desde 2          # Ejecuta desde el paso 2 hasta el final
  python ejecutar_pipeline.py --hasta 2          # Ejecuta hasta el paso 2
        """
    )
    
    parser.add_argument('--paso', type=int, choices=[1, 2, 3],
                       help='Ejecuta solo un paso específico (1, 2 o 3)')
    parser.add_argument('--desde', type=int, choices=[1, 2, 3],
                       help='Ejecuta desde este paso hasta el final')
    parser.add_argument('--hasta', type=int, choices=[1, 2, 3],
                       help='Ejecuta hasta este paso')
    
    args = parser.parse_args()
    
    # Determinar qué pasos ejecutar
    if args.paso:
        pasos = [args.paso]
    elif args.desde and args.hasta:
        if args.desde > args.hasta:
            print("Error: --desde debe ser menor o igual que --hasta")
            sys.exit(1)
        pasos = list(range(args.desde, args.hasta + 1))
    elif args.desde:
        pasos = list(range(args.desde, 4))
    elif args.hasta:
        pasos = list(range(1, args.hasta + 1))
    else:
        pasos = [1, 2, 3]
    
    print(f"\nPasos a ejecutar: {pasos}")
    
    # Inicializar modelos solo si son necesarios
    kw_model = None
    embed_model = None
    
    if 1 in pasos:
        print("\nInicializando modelo KeyBERT...")
        kw_model = KeyBERT(EMBEDDING_MODEL)
    
    if 3 in pasos:
        print("\nInicializando modelo SentenceTransformer...")
        embed_model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Ejecutar pasos
    datos = None
    for paso in pasos:
        resultado = ejecutar_paso(paso, kw_model, embed_model)
        if resultado is None and paso != 3:  # paso 3 no retorna datos
            print(f"\nError en el paso {paso}. Deteniendo ejecución.")
            sys.exit(1)
        if resultado is not None:
            datos = resultado
    
    # Resumen final
    print("\n" + "="*70)
    print("EJECUCIÓN COMPLETADA")
    print("="*70)
    
    if datos:
        print(f"\nTotal de fragmentos: {len(datos)}")
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProceso interrumpido por el usuario.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)