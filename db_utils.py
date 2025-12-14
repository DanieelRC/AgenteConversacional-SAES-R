import mysql.connector
from mysql.connector import pooling
import datetime
from typing import Optional, Dict, Any
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuración del pool de conexión a la base de datos MySQL
db_pool = pooling.MySQLConnectionPool(
    pool_name="saes_pool",
    pool_size=32,
    pool_reset_session=True,
    host=os.getenv("DB_HOST", "localhost"),
    user=os.getenv("DB_USER", "root"),
    password=os.getenv("DB_PASSWORD", "root"),
    database=os.getenv("DB_NAME", "SAES"),
    auth_plugin=os.getenv("DB_AUTH_PLUGIN", "mysql_native_password")
)

def _get_db_connection():
    """Obtiene una conexión del pool."""
    try:
        return db_pool.get_connection()
    except mysql.connector.Error as err:
        logging.error(f"Error al obtener conexión del pool: {err}")
        return None

def obtener_datos_usuario(boleta: str) -> Optional[Dict[str, Any]]:
    """Obtiene todos los datos académicos de un alumno."""
    conn = None
    cursor = None
    try:
        if not boleta or not isinstance(boleta, str):
            return None

        conn = _get_db_connection()
        if not conn: return None
        cursor = conn.cursor(dictionary=True)

        # 1. Datos Personales y Académicos de Estudiante
        # Se une datos_personales con estudiante
        cursor.execute("""
            SELECT 
                dp.id AS boleta,
                dp.nombre,
                dp.ape_paterno,
                dp.ape_materno,
                dp.email,
                dp.carrera,
                dp.telefono,
                CONCAT_WS(', ', dp.calle, CONCAT('Núm. ', dp.num_exterior), 
                          dp.colonia, dp.delegacion, dp.ciudad, CONCAT('CP ', dp.codigo_postal)) AS direccion_completa,
                e.promedio,
                e.creditos_disponibles,
                e.estado_academico
            FROM datos_personales AS dp
            JOIN estudiante AS e ON dp.id = e.id_usuario
            WHERE dp.id = %s;
        """, (boleta,))
        info = cursor.fetchone()
        if not info:
            return None
        
        # 2. Resumen Kardex (Situación y Semestres Restantes)
        cursor.execute("""
            SELECT promedio, situacion_academica, semestres_restantes
            FROM kardex
            WHERE id_alumno = %s
            ORDER BY id DESC
            LIMIT 1;
        """, (boleta,))
        kardex_resumen = cursor.fetchone() or {}

        # 3. UA Aprobadas
        cursor.execute("""
            SELECT 
                ua.unidad_aprendizaje AS materia,
                ua.calificacion_final AS calificacion,
                ua.semestre,
                ua.metodo_aprobado,
                ua.periodo,
                ua.fecha
            FROM kardex AS k
            JOIN ua_aprobada AS ua ON k.id = ua.id_kardex
            WHERE k.id_alumno = %s
            ORDER BY ua.fecha DESC;
        """, (boleta,))
        materias_aprobadas_raw = cursor.fetchall() or []
        
        # 4. UA Reprobadas
        cursor.execute("""
            SELECT 
                mr.id AS id_reprobada, 
                ua.nombre AS materia,
                mr.periodos_restantes,
                mr.recurse,
                mr.estado_actual
            FROM materia_reprobada AS mr
            JOIN unidad_de_aprendizaje AS ua ON mr.id_ua = ua.id
            WHERE mr.id_estudiante = %s;
        """, (boleta,))
        materias_reprobadas_raw = cursor.fetchall() or []

        # 5. Materias Inscritas (Actuales) con Profesores y Horarios Detallados
        cursor.execute("""
            SELECT DISTINCT
                u.nombre AS materia,
                g.nombre AS grupo,
                g.turno,
                u.credito AS credito,
                CONCAT_WS(' ', dp2.nombre, dp2.ape_paterno, dp2.ape_materno) AS profesor_nombre,
                u.semestre,
                GROUP_CONCAT(
                    CONCAT(d.dia, ' ', d.hora_ini, '-', d.hora_fin) 
                    ORDER BY FIELD(d.dia, 'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'), d.hora_ini
                    SEPARATOR ', '
                ) AS horario_detallado
            FROM horario AS h
            JOIN mat_inscritos AS mi ON h.id = mi.id_horario
            JOIN grupo AS g ON mi.id_grupo = g.id
            JOIN unidad_de_aprendizaje AS u ON g.id_ua = u.id
            JOIN datos_personales AS dp2 ON g.id_prof = dp2.id
            LEFT JOIN distribucion AS d ON g.id = d.id_grupo
            WHERE h.id_alumno = %s
            GROUP BY u.nombre, g.nombre, g.turno, u.credito, profesor_nombre, u.semestre;
        """, (boleta,))
        materias_inscritas_raw = cursor.fetchall() or []
        
        # 6. Reinscripción (Ventana activa)
        cursor.execute("""
            SELECT 
                EXISTS(
                    SELECT 1 
                    FROM inscripcion 
                    WHERE id_alumno = %s 
                      AND NOW() BETWEEN fecha_hora_in AND fecha_hora_cad
                ) AS reinscripcion_activa,
                MAX(fecha_hora_cad) AS inscripcion_caduca
            FROM inscripcion
            WHERE id_alumno = %s;
        """, (boleta, boleta))
        reinsc = cursor.fetchone() or {"reinscripcion_activa": 0, "inscripcion_caduca": None}
        
        # 7. Fechas relevantes
        cursor.execute("""
            SELECT * FROM fechas_relevantes ORDER BY inicio_semestre DESC LIMIT 1;
        """)
        fechas_raw = cursor.fetchone() or {}

        # --- Formateo de Textos y Fechas ---
        
        # Materias Aprobadas
        materias_aprobadas_txt = [
            f"- {m['materia']} (Calif: {m['calificacion']}, {m['metodo_aprobado']})"
            for m in materias_aprobadas_raw
        ]
        
        # Materias Reprobadas (Kardex Detalle)
        materias_reprobadas_txt = [
            f"- {m['materia']} (Recursos restantes: {m['periodos_restantes']}, Estado: {m['estado_actual']})"
            for m in materias_reprobadas_raw
        ]
        
        # Materias Inscritas (Horario/Grupos) con Horarios Detallados
        materias_inscritas_txt = []
        for m in materias_inscritas_raw:
            horario_raw = m.get('horario_detallado', '')
            # Formatear horario de manera más clara para el LLM
            if horario_raw:
                # Convertir "Lunes 7:00-8:30, Martes 10:00-11:30" a formato más legible
                horarios_list = horario_raw.split(', ')
                horario_formateado = []
                for h in horarios_list:
                    partes = h.split(' ')
                    if len(partes) == 2:
                        dia = partes[0]
                        horas = partes[1].split('-')
                        if len(horas) == 2:
                            horario_formateado.append(f"{dia} de {horas[0]} a {horas[1]}")
                        else:
                            horario_formateado.append(h)
                    else:
                        horario_formateado.append(h)
                horario_texto = ', '.join(horario_formateado)
            else:
                horario_texto = 'Sin horario asignado'
            
            materias_inscritas_txt.append(
                f"- {m['materia']} (Gpo: {m['grupo']}, Turno: {m['turno']}, Prof: {m['profesor_nombre']})\n"
                f"  Horario: {horario_texto}"
            )
        
        # Fechas
        fechas_dict = {}
        for k, v in fechas_raw.items():
            if isinstance(v, (datetime.datetime, datetime.date)):
                fechas_dict[k] = v.strftime("%Y-%m-%d %H:%M:%S")
            else:
                fechas_dict[k] = str(v) if v else "N/A"
        
        caduca_val = reinsc.get("inscripcion_caduca")
        caduca_str = caduca_val.strftime("%Y-%m-%d %H:%M:%S") if isinstance(caduca_val, (datetime.datetime, datetime.date)) else "N/A"
        
        semestre_actual = max((m.get("semestre") or 0) for m in materias_inscritas_raw) if materias_inscritas_raw else None
        
        return {
            "boleta": info["boleta"],
            "nombre": f"{info['nombre']} {info['ape_paterno']} {info['ape_materno']}",
            "correo": info["email"],
            "telefono": info.get("telefono", "N/A"),
            "direccion_completa": info.get("direccion_completa", "N/A"),
            "carrera": info["carrera"],
            "promedio": info.get("promedio"),
            "creditos_disponibles": info.get("creditos_disponibles"),
            "estado_academico": info.get("estado_academico"),
            
            # Kardex
            "situacion_kardex": kardex_resumen.get("situacion_academica"),
            "semestres_restantes": kardex_resumen.get("semestres_restantes"),
            "total_materias_aprobadas": len(materias_aprobadas_raw),
            "materias_aprobadas_texto": "\n".join(materias_aprobadas_txt) or "Sin materias aprobadas registradas",
            "materias_reprobadas_texto": "\n".join(materias_reprobadas_txt) or "Sin materias reprobadas registradas",
            
            # Inscripción
            "semestre_actual": semestre_actual,
            "total_materias_inscritas": len(materias_inscritas_raw),
            "materias_inscritas_texto": "\n".join(materias_inscritas_txt) or "Sin materias inscritas actualmente",
            "reinscripcion_activa": bool(reinsc.get("reinscripcion_activa", 0)),
            "inscripcion_caduca": caduca_str,

            "fechas_semestre": fechas_dict
        }

    except mysql.connector.Error as err:
        logging.error(f"Error MySQL en obtener_datos_usuario: {err}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"Error general en obtener_datos_usuario: {e}", exc_info=True)
        return None
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()


def obtener_datos_profesor(id_profesor: str) -> Optional[Dict[str, Any]]:
    """Obtiene todos los datos académicos de un profesor."""
    conn = None
    cursor = None
    try:
        if not id_profesor or not isinstance(id_profesor, str):
            return None

        conn = _get_db_connection()
        if not conn: return None
        cursor = conn.cursor(dictionary=True)

        # 1. Datos Personales y Calificación (usando datos_personales.calificacion)
        cursor.execute("""
            SELECT 
                dp.id AS id_profesor,
                dp.nombre,
                dp.ape_paterno,
                dp.ape_materno,
                dp.email,
                dp.telefono,
                dp.grado,
                dp.calificacion
            FROM datos_personales AS dp
            WHERE dp.id = %s AND dp.tipo_usuario = 'profesor';
        """, (id_profesor,))
        info = cursor.fetchone()
        if not info:
            return None

        # 2. Grupos Impartidos
        cursor.execute("""
            SELECT 
                ua.nombre AS materia, 
                g.nombre AS grupo, 
                g.turno,
                g.cupo
            FROM grupo AS g
            JOIN unidad_de_aprendizaje AS ua ON g.id_ua = ua.id
            WHERE g.id_prof = %s;
        """, (id_profesor,))
        grupos_raw = cursor.fetchall() or []

        # 3. Reseñas y Calificación Promedio Real (usando la tabla contador)
        cursor.execute("""
            SELECT 
                registrados AS total_resenas,
                suma / registrados AS promedio_calculado
            FROM contador
            WHERE id_profesor = %s;
        """, (id_profesor,))
        stats_contador = cursor.fetchone() or {}

        # 4. Últimos 5 Comentarios de Reseñas
        cursor.execute("""
            SELECT comentarios, calificacion, fecha
            FROM resena
            WHERE id_profesor = %s
            ORDER BY fecha DESC
            LIMIT 5;
        """, (id_profesor,))
        ultimos_comentarios_raw = cursor.fetchall() or []

        # 5. Fechas relevantes
        cursor.execute("""
            SELECT * FROM fechas_relevantes ORDER BY inicio_semestre DESC LIMIT 1;
        """)
        fechas_raw = cursor.fetchone() or {}

        # --- Formateo de Textos y Fechas ---
        
        # Calificación
        calificacion_promedio = stats_contador.get("promedio_calculado") if stats_contador.get("registrados", 0) > 0 else info.get("calificacion", 0.0)
        
        # Grupos Impartidos
        grupos_txt = [
            f"- {g['materia']} (Gpo: {g['grupo']}, Turno: {g['turno']}, Cupo: {g['cupo']})"
            for g in grupos_raw
        ]
        
        # Comentarios
        comentarios_txt = []
        for c in ultimos_comentarios_raw:
            fecha_str = c['fecha'].strftime("%Y-%m-%d") if isinstance(c['fecha'], (datetime.datetime, datetime.date)) else "N/A"
            comentarios_txt.append(f"- \"{c['comentarios']}\" (Calif: {c['calificacion']}, Fecha: {fecha_str})")
        
        # Fechas
        fechas_dict = {}
        for k, v in fechas_raw.items():
            if isinstance(v, (datetime.datetime, datetime.date)):
                fechas_dict[k] = v.strftime("%Y-%m-%d %H:%M:%S")
            else:
                fechas_dict[k] = str(v) if v else "N/A"

        return {
            "id_profesor": info["id_profesor"],
            "nombre": f"{info['nombre']} {info['ape_paterno']} {info['ape_materno']}",
            "correo": info["email"],
            "telefono": info.get("telefono", "N/A"),
            "grado": info.get("grado", "N/A"),
            "calificacion_promedio": calificacion_promedio,
            "total_resenas": stats_contador.get("registrados", 0),
            
            # Grupos
            "grupos_texto": "\n".join(grupos_txt) or "Sin grupos asignados para este semestre.",
            
            # Reseñas
            "ultimos_comentarios": "\n".join(comentarios_txt) or "Sin comentarios recientes.",
            
            "fechas_semestre": fechas_dict
        }

    except mysql.connector.Error as err:
        logging.error(f"Error MySQL en obtener_datos_profesor: {err}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"Error general en obtener_datos_profesor: {e}", exc_info=True)
        return None
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()