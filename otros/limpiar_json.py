import json
import re

def limpiar_texto(texto):
    """
    Limpia el texto eliminando ambigüedades léxicas:
    - Palabras cortadas con guión al final de línea
    - Espacios en blanco innecesarios
    - Múltiples espacios consecutivos
    """
    if not texto:
        return texto
    
    # Eliminar palabras cortadas con guión seguido de espacio o salto de línea
    # Ejemplo: "des- tacar" -> "destacar"
    texto = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', texto)
    
    # Eliminar guiones al final de palabras seguidos de espacio
    # Ejemplo: "palabra- " -> "palabra"
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

def limpiar_json(archivo_entrada, archivo_salida=None):
    """
    Lee un archivo JSON, limpia el contenido y guarda el resultado.
    
    Args:
        archivo_entrada: Ruta del archivo JSON a limpiar
        archivo_salida: Ruta donde guardar el resultado (si es None, sobreescribe el original)
    """
    print(f"Leyendo archivo: {archivo_entrada}")
    
    # Leer el archivo JSON
    with open(archivo_entrada, 'r', encoding='utf-8') as f:
        datos = json.load(f)
    
    print(f"Total de fragmentos: {len(datos)}")
    
    # Contador de cambios
    cambios_texto = 0
    cambios_palabras = 0
    
    # Limpiar cada fragmento
    for i, fragmento in enumerate(datos):
        # Limpiar el texto
        texto_original = fragmento.get('texto', '')
        texto_limpio = limpiar_texto(texto_original)
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
            print(f"Procesados {i + 1} fragmentos...")
    
    print(f"\nEstadísticas de limpieza:")
    print(f"- Textos modificados: {cambios_texto}")
    print(f"- Grupos de palabras clave modificados: {cambios_palabras}")
    
    # Determinar archivo de salida
    if archivo_salida is None:
        archivo_salida = archivo_entrada
    
    # Guardar el archivo limpio
    print(f"\nGuardando archivo limpio en: {archivo_salida}")
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)
    
    print("¡Limpieza completada!")
    
    return datos

def mostrar_ejemplos(archivo_entrada, num_ejemplos=5):
    """
    Muestra ejemplos de palabras que serían limpiadas sin modificar el archivo.
    """
    with open(archivo_entrada, 'r', encoding='utf-8') as f:
        datos = json.load(f)
    
    print("=== EJEMPLOS DE LIMPIEZA ===\n")
    
    ejemplos_encontrados = 0
    for fragmento in datos:
        if ejemplos_encontrados >= num_ejemplos:
            break
        
        # Buscar palabras con problemas
        palabras_problematicas = []
        for palabra in fragmento.get('palabras_clave', []):
            if palabra.endswith('-') or palabra.endswith(' ') or '- ' in palabra:
                palabras_problematicas.append(palabra)
        
        if palabras_problematicas:
            ejemplos_encontrados += 1
            print(f"Fragmento: {fragmento['fragmento_id']}")
            print(f"Palabras problemáticas encontradas:")
            for palabra in palabras_problematicas:
                palabra_limpia = palabra.rstrip('- ').strip()
                print(f"  '{palabra}' → '{palabra_limpia}'")
            print()
    
    if ejemplos_encontrados == 0:
        print("No se encontraron ejemplos claros en las primeras revisiones.")

if __name__ == "__main__":
    archivo = "reglamentos_ipn.json"
    
    print("=" * 60)
    print("SCRIPT DE LIMPIEZA DE AMBIGÜEDADES LÉXICAS EN JSON")
    print("=" * 60)
    print()
    
    # Primero mostrar algunos ejemplos
    print("1. Mostrando ejemplos de lo que se va a limpiar...\n")
    mostrar_ejemplos(archivo)
    
    # Preguntar si desea continuar
    respuesta = input("\n¿Desea proceder con la limpieza del archivo? (s/n): ").lower()
    
    if respuesta == 's':
        # Crear backup
        archivo_backup = archivo.replace('.json', '_backup.json')
        print(f"\nCreando backup en: {archivo_backup}")
        with open(archivo, 'r', encoding='utf-8') as f_in:
            with open(archivo_backup, 'w', encoding='utf-8') as f_out:
                f_out.write(f_in.read())
        
        # Limpiar el archivo
        print()
        limpiar_json(archivo)
        print(f"\nBackup guardado en: {archivo_backup}")
    else:
        print("\nOperación cancelada.")
