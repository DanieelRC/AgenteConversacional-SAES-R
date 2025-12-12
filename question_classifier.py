import re
import unicodedata
from typing import Dict, Tuple, Optional, Callable

DEFINICIONES = {
    "Academia": "Órgano constituido por profesores que tiene la finalidad de proponer, analizar, opinar, estructurar y evaluar el proceso educativo.",
    "Actividades complementarias": "Aquéllas que contribuyen a la formación integral del alumno y que no necesariamente forman parte del programa académico en el que se encuentra inscrito.",
    "Alumno": "A la persona inscrita en algún programa académico que se imparta en cualquier nivel educativo y modalidad educativa que ofrece el Instituto Politécnico Nacional.",
    "Alumno en movilidad": "Aquél en situación escolar regular que cursa unidades de aprendizaje, desarrolla actividades de investigación o complementarias en una institución educativa, de investigación o del sector productivo, nacional o extranjera, de conformidad con la normatividad institucional y, en su caso, con los convenios correspondientes.",
    "Alumno visitante": "Aquél de otra institución educativa nacional o extranjera que cursa unidades de aprendizaje o desarrolla actividades de investigación o complementarias en el Instituto, de conformidad con la normatividad institucional y de acuerdo a los convenios correspondientes, mismo que será considerado como alumno durante el tiempo que se encuentre inscrito en dichas unidades o actividades.",
    "Ambientes de aprendizaje": "A los espacios y recursos disponibles para la intermediación en la adquisición y generación del conocimiento.",
    "Calendario académico": "A la programación que define los tiempos en los cuales se realizan anualmente las actividades académicas y de gestión escolar, en las diversas modalidades educativas que imparte el Instituto Politécnico Nacional.",
    "Carga máxima en créditos": "Al resultado de dividir el número total de créditos del programa académico entre el número de periodos escolares de la duración mínima del plan de estudio.",
    "Carga media en créditos": "Al resultado de dividir el número total de créditos del programa académico entre el número de periodos escolares de la duración establecida en el plan de estudio.",
    "Carga mínima en créditos": "Al resultado de dividir el número total de créditos del programa académico entre el número de periodos escolares de la duración máxima del plan de estudio.",
    "Ciclo escolar": "Al lapso anual que define el Calendario Académico del Instituto Politécnico Nacional.",
    "Comisión de Situación Escolar": "Al órgano colegiado que emana de los Consejos Técnicos Consultivos Escolares, del Consejo General Consultivo, o es reconocido por éste y se encarga de dictaminar los asuntos derivados de la situación escolar, en los términos de la normatividad aplicable.",
    "Cooperación académica": "A las acciones conjuntas entre dos o más instituciones nacionales o extranjeras, en las que participan alumnos, profesores, investigadores y personal administrativo, relacionadas con docencia, investigación, extensión de los conocimientos, difusión de la cultura, promoción del deporte y apoyo a la administración, gestión y dirección, en el marco de un proyecto o programa.",
    "Crédito": "A la unidad de reconocimiento académico que mide y cuantifica las actividades de aprendizaje contempladas en un plan de estudio; es universal, transferible entre programas académicos y equivalente al trabajo académico del alumno.",
    "Dirección de Coordinación": "A las direcciones de educación media superior, de educación superior, de posgrado, de educación continua, de formación en lenguas extranjeras, de administración escolar, así como la coordinación de cooperación académica.",
    "Egreso": "Al proceso mediante el cual el alumno concluye sus estudios y acredita la totalidad del programa académico en el que estuvo inscrito.",
    "Evaluación a título de suficiencia": "A la que comprende el total de los contenidos del programa de estudios y que el alumno podrá presentar cuando no haya acreditado de manera ordinaria o extraordinaria alguna unidad de aprendizaje.",
    "ETS": "A la que comprende el total de los contenidos del programa de estudios y que el alumno podrá presentar cuando no haya acreditado de manera ordinaria o extraordinaria alguna unidad de aprendizaje.",
    "Evaluación de saberes previamente adquiridos": "A la que permite acreditar unidades de aprendizaje sin haberlas cursado. Su aplicación se sujetará a lo descrito en el plan y programa de estudios, y a los lineamientos aplicables.",
    "ESPA": "A la que permite acreditar unidades de aprendizaje sin haberlas cursado. Su aplicación se sujetará a lo descrito en el plan y programa de estudios, y a los lineamientos aplicables.",
    "Evaluación extraordinaria": "A la que comprende el total de los contenidos del programa de estudios y que el alumno podrá presentar voluntariamente, dentro del mismo periodo escolar, una vez que cursó la unidad de aprendizaje y no haya obtenido un resultado aprobatorio, o bien, si habiéndola acreditado, desea mejorar su calificación.",
    "Evaluación ordinaria": "A la que se presenta con fines de acreditación durante el periodo escolar y considera las evidencias de aprendizaje señaladas en el programa de estudios.",
    "Expediente Académico": "Al documento que contiene la información y el historial académico del alumno.",
    "Flexibilidad": "Característica del plan de estudio que permite al alumno definir su trayectoria escolar dentro del marco de la normatividad aplicable.",
    "Ingreso": "Al proceso a través del cual el aspirante a incorporarse como alumno o usuario de servicios educativos complementarios cumple con todos los requisitos de admisión establecidos para cualquier programa académico o servicio educativo que ofrece el Instituto Politécnico Nacional.",
    "Instituto": "Al Instituto Politécnico Nacional.",
    "Mapa curricular": "A la representación gráfica de las unidades de aprendizaje que conforman un plan de estudio.",
    "Modalidad educativa": "A la forma en que se organizan, distribuyen y desarrollan los planes y programas de estudio para su impartición.",
    "Movilidad académica": "Al proceso que permita al alumno, en situación escolar regular, participar en programas académicos o desarrollar actividades académicas complementarias en instituciones nacionales o extranjeras con las que el Instituto tenga convenio para tal fin o formen parte de un programa académico reconocido que incluya tal movilidad.",
    "Nivel educativo": "A cada una de las etapas en las que se estructuran los estudios que ofrece el Instituto: medio superior, superior y posgrado.",
    "Periodo escolar": "Al lapso señalado en el calendario académico para cursar unidades de aprendizaje de un programa académico.",
    "Plan de estudio": "A la estructura curricular que se deriva de un programa académico y que permite cumplir con los propósitos de formación general, la adquisición de conocimientos y el desarrollo de capacidades correspondientes a un nivel y modalidad educativa.",
    "Programa académico": "Al conjunto organizado de elementos necesarios para generar, adquirir y aplicar el conocimiento en un campo específico; así como para desarrollar habilidades, actitudes y valores en el alumno, en diferentes áreas del conocimiento.",
    "Programa académico en red": "Al que desarrollan e imparten conjuntamente varias unidades académicas del Instituto o con otras instituciones con las que se tenga convenio.",
    "Programa de estudios": "A los contenidos formativos de una unidad de aprendizaje contemplada en un plan de estudio; especifica los objetivos a lograr por los alumnos en un periodo escolar; establece la carga horaria, número de créditos, tipos de espacios, ambientes y actividades de aprendizaje, prácticas escolares, bibliografía, plan de evaluación y programa sintético.",
    "Trayectoria escolar": "Al proceso a través del cual el alumno construye su formación con base en un plan de estudio.",
    "Tutor": "Al personal académico asignado para acompañar, orientar y asesorar al alumno en su trayectoria escolar con la finalidad de que concluya satisfactoriamente sus estudios.",
    "Usuario de servicios educativos complementarios": "A la persona registrada en cualquiera de los programas que ofrece el Instituto en materia de capacitación, actualización técnica y profesional, formación empresarial, educación continua o formación de capacidades a lo largo de la vida y lenguas extranjeras, entre otros.",
    "Dictamen": "Al proceso formal y oficial mediante el cual un alumno, generalmente con una situación académica irregular, solicita un dictamen a las autoridades escolares competentes. Este dictamen es una resolución que le permite continuar o regularizar su trayectoria académica a pesar de haber incurrido en alguna falta a la normativa, como tener materias reprobadas o haber excedido el tiempo reglamentario para finalizar sus estudios. Si el dictamen es favorable, le otorga la autorización para reinscribirse, presentar evaluaciones a título de suficiencia (ETS), o cualquier otra acción necesaria para recuperar su calidad de estudiante y proseguir su formación.",
    "Dictaminado": "Al estado en el que se encuentra un alumno que ha solicitado un dictamen a las autoridades escolares competentes."
}

def compile_patterns(pattern_dict: Dict[str, list]) -> Dict[str, list]:
    """Compila todas las expresiones regulares del diccionario."""
    compiled = {}
    for key, patterns in pattern_dict.items():
        compiled[key] = [re.compile(p, re.IGNORECASE) for p in patterns]
    return compiled


def compile_list(pattern_list: list) -> list:
    """Compila una lista de patrones."""
    return [re.compile(p, re.IGNORECASE) for p in pattern_list]

def normalize_for_regex(text: str) -> str:
    """Normaliza texto para crear regex"""
    text = text.lower()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    text = re.escape(text)
    text = text.replace(r"\ ", r"\s+")
    text = text.replace(" ", r"\s+")
    return text

_ANSWER_BUILDERS: Dict[str, Callable[[Dict], str]] = {}

class QuestionClassifier:
    """Clasifica preguntas en directas (BD) o complejas (LLM)."""
    
    DIRECT_PATTERNS_RAW = {
        "horario": [
            r"hora", r"horario", r"cual.*mi.*horario", r"que.*horario", r"a que.*hora",
            r"a que.*h", r"mi.*horario", r"horario.*clases", r"mis.*clases.*a.*que.*hora",
            r"horario.*actual", r"a.*que.*entro", r"a.*que.*hora.*salgo", 
            r"rol.*clases", r"rol.*horario", r"a.*que.*toca", r"q.*hora", 
        ],
        "materias_inscritas": [
            r"que.*materias.*inscrit", r"materias.*curs", r"mis\s+materias", r"materias.*actual",
            r"cursos.*actual", r"que.*llevo", r"mis.*cursos", r"cuantas.*materias.*tengo",
            r"que.*voy.*a.*cursar", r"cuales.*son.*mis.*unidades.*de.*aprendizaje", 
            r"que.*mats.*llevo", r"mats.*inscritas", r"mis.*ua", r"cuales.*son.*mis.*materias", 
        ],
        "promedio": [
            r"cual.*promedio", r"^promedio$", r"mi.*calificacion", r"prom.*tengo", r"promedito",
            r"cual.*es.*mi.*prom", r"calif.*general", r"mi.*promedio.*actual", r"nota.*media", 
            r"q.*promedio", r"q.*calif", r"mi.*prome", r"mi.*promedios", 
        ],
        "creditos": [
            r"cuantos.*creditos", r"^creditos$", r"cuantos.*creditos.*tengo", r"total.*creditos",
            r"mis.*creditos", r"creditos.*totales", r"cuantos.*creditos.*acumulo", 
            r"cuantos.*creditos.*llevo", r"cantidad.*creditos", r"creditos.*en.*total", 
        ],
        "estado": [
            r"estado.*academico", r"situacion.*academica", r"como.*voy.*escuela", r"mi.*situacion",
            r"como.*esta.*mi.*estado", r"mi.*estatus.*academico", r"estatus.*escolar", 
            r"situacion.*actual", r"como.*va.*mi.*carrera", r"mi.*situacion.*escolar", 
        ],
        "materias_aprobadas": [
            r"materias.*aprob", r"kardex", r"historial.*academico", r"kardexcito",
            r"cuantas.*pase", r"que.*materias.*pase", r"mi.*historial", r"mi.*kardex",
            r"materias.*acreditadas", r"calificaciones.*finales", r"record.*academico", 
            r"mats.*pasadas", r"mats.*aprobadas.*tengo", r"mi.*kardez", r"kardex.*completo", 
        ],
        "carrera": [
            r"cual.*carrera", r"en que.*estudio", r"mi.*carrera", r"que.*estoy.*estudiando",
            r"programa.*academico", r"cual.*es.*mi.*programa", r"en.*que.*programa.*estoy", 
            r"q.*carrera", r"mi.*licenciatura", r"mi.*ingenieria", r"mi.*carrera.*es", 
        ],
        "semestre": [
            r"en que.*semestre", r"^semestre$", r"que.*semestre.*curso", r"semestre.*voy",
            r"en que.*voy", r"mi.*semestre.*actual", r"nivel.*cursando", 
            r"q.*semestre", r"q.*nivel", r"en.*que.*perido.*estoy", r"en.*el.*semestre", 
        ],
        "datos_personales": [
            r"cual.*boleta", r"mi.*boleta", r"cual.*correo", r"mi.*correo", r"mi.*email",
            r"cual.*nombre", r"mi.*nombre", r"cual.*telefono", r"mi.*telefono", r"cual.*direccion",
            r"mi.*direccion", r"la.*boleta", r"mis.*datos", r"mi.*tel", r"mi.*dir",
            r"mi.*numero.*de.*boleta", r"datos.*de.*contacto", r"domicilio.*registrado",    
            r"num.*boleta", r"mi.*mail", r"mi.*tel\b", r"mi.*dire", 
        ],
        "inscripcion_info": [
            r"cuando.*caduca.*inscripcion", r"puedo.*reinscrib", r"reinscripcion.*activa",
            r"hasta cuando.*reinscrib", r"cuando.*vence.*inscripcion", r"hay.*reinscripcion",
            r"hasta.*cuando.*tengo.*reinscribirme", r"cuando.*termina.*mi.*inscripcion",
            r"fecha.*limite.*reinscripcion", r"se.*me.*pasa.*la.*reinscripcion", 
            r"cuando.*es.*la.*reinscrip", r"reinscripcion.*vence", r"reinscribirme.*es.*posible", 
        ],
        "creditos_detalle": [
            r"creditos.*disponibles", r"creditos.*cursando", r"creditos.*inscritos",
            r"cuantos.*creditos.*falta", r"creditos.*actuales", r"creditos.*maximos.*minimos", 
            r"creditos.*cargados", r"creditos.*puedo.*llevar", r"carga.*creditos", 
        ],
        "programa_info": [
            r"cuantos.*semestres.*dura", r"duracion.*programa", r"cuantos.*semestres.*quedan",
            r"cuantos.*semestres.*faltan", r"cuanto.*dura.*carrera", r"tiempo.*me.*queda",
            r"cuanto.*tiempo.*tengo.*para.*acabar", r"duracion.*de.*mi.*plan", 
            r"cuanto.*falta.*acabar", r"cuantos.*semestres.*son", r"duracion.*total", 
        ],
        "conteo_materias": [
            r"cuantas.*materias.*cursando", r"cuantas.*materias.*inscrit", r"cuantas.*materias.*aprob",
            r"cuantas.*materias.*llevo", r"cuantas.*tengo.*inscritas", r"cuantas.*llevo.*pasadas",
            r"numero.*de.*materias.*cursadas", r"total.*de.*materias", 
            r"num.*mats.*llevo", r"conteo.*materias", r"cuantas.*mats.*tengo", 
        ],
        "kardex_info": [
            r"situacion.*kardex", r"que.*dice.*kardex", r"como.*esta.*kardex", r"mi.*estado.*kardex",
            r"como.*ando.*kardex", r"que.*informacion.*tiene.*mi.*kardex", r"datos.*del.*kardex", 
            r"info.*kardex", r"como.*saco.*mi.*kardex", r"ver.*kardex", 
        ],
        "turno_info": [
            r"en que.*turno", r"clases.*manana", r"clases.*tarde", r"turno.*soy",
            r"en.*que.*turno.*me.*toco", r"turno.*asignado", r"turno.*de.*estudios", 
            r"es.*matutino.*o.*vespertino", r"mi.*turno", r"clases.*en.*la.*tarde", 
        ],
        "profesores_info": [
            r"en que.*grupo", r"quienes.*profesores", r"que.*profesores.*tengo",
            r"quien.*me.*da.*clases", r"mis.*profes", r"maestros.*tengo",
            r"lista.*de.*profesores", r"mis.*docentes", r"quien.*me.*imparte", 
            r"q.*profes", r"profesor.*de.*la.*materia", r"quien.*me.*toca", 
        ],
        "fechas_semestre": [
            r"cuando.*empieza.*semestre", r"cuando.*termina.*semestre", r"inicio.*semestre",
            r"fin.*semestre", r"fecha.*inicio.*clases", r"cuando.*acaba.*semestre",
            r"calendario.*escolar", r"periodo.*semestral", r"fechas.*del.*ciclo.*escolar", 
            r"cuando.*inicia.*clases", r"fecha.*fin.*semestre", r"calendario.*inicio", 
        ],
        "fechas_parciales": [
            r"cuando.*parcial", r"fecha.*examen", r"cuando.*examen", r"cuando.*son.*parciales",
            r"fechas.*parcial", r"cuando.*presento.*examenes", r"calendario.*de.*evaluaciones", 
            r"fechas.*de.*parciales", r"examen.*de.*primer.*parcial", r"fechas.*de.*evaluacion", 
        ],
        "fechas_ets": [
            r"cuando.*ets", r"subir.*documentos", r"evaluacion.*profesores",
            r"fechas.*ets", r"cuando.*aplico.*ets", r"examen.*titulo.*suficiencia",
            r"fechas.*examen.*a.*titulo", r"cuando.*es.*la.*evaluacion.*ets", 
            r"fecha.*limite.*subir.*doc", r"evaluacion.*ets", r"cuando.*califican.*ets", 
        ],
        "profesor_grupos": [
            r"mis.*grupos", r"grupos.*imparto", r"clases.*que.*doy", r"mis.*clases",
            r"distribucion.*clases", r"horario.*clases", r"materias.*doy",
            r"que.*grupos.*tengo", r"mis.*grupos.*asignados", r"lista.*de.*grupos", 
            r"grupos.*que.*atiendo", r"mi.*carga.*academica", r"mis.*clases.*asignadas", 
        ],
        "profesor_calificacion": [
            r"mi.*calificacion", r"calificacion.*tengo", r"promedio.*resenas",
            r"evaluacion.*desempeno", r"cual.*promedio.*profesor", r"calif.*alumno",
            r"mi.*puntaje.*promedio", r"nota.*general.*profesor", r"mi.*promedio.*de.*evaluacion", 
        ],
        "profesor_resenas": [
            r"mis.*resenas", r"comentarios.*alumnos", r"que.*dicen.*de.*mi",
            r"comentarios.*recibi", r"resenas.*recientes", r"opiniones.*de.*alumnos", 
            r"comentarios.*sobre.*mi.*clase", r"ultimas.*resenas", r"calificaciones.*alumno.*profesor", 
        ],
        "profesor_fechas": [
            r"fechas.*importantes", r"calendario", r"subo.*calificaciones",
            r"fecha.*registro.*calificaciones", r"limite.*subir.*calif",
            r"cuando.*tengo.*que.*entregar.*calificaciones", r"fechas.*limite.*registro.*notas", 
            r"cuando.*se.*suben.*calif", r"registro.*notas.*parcial", r"fecha.*corte.*calificaciones", 
        ],
    }
    
    COMPLEX_PATTERNS_RAW = [
        r"puedo.*inscrib", r"requisito", r"como.*puedo", r"reglamento", r"\bbaja\b", r"titulacion",
        r"extraordinario", r"suficiencia", r"como.*dar.*baja", r"como.*me.*titulo",
        r"que.*necesito.*titularme", r"cuando.*pido.*baja", r"que.*pasa.*si.*repruebo",
        r"tramite.*de.*baja", r"baja.*temporal.*definitiva", r"que.*es.*el.*reglamento", 
        r"que.*hago.*si.*repruebo", r"opciones.*de.*titulacion", r"reglamento.*escolar", 
        r"como.*inscribo.*materias", r"reglas.*del.*ipn", r"cuantos.*extraordinarios.*puedo", 
    ]

    for term in DEFINICIONES.keys():
        if len(term) > 50: continue # Ignorar claves muy largas (como el header)
        term_norm = normalize_for_regex(term)
        key = f"definicion_{term}"
        DIRECT_PATTERNS_RAW[key] = [
            f"que.*es.*{term_norm}",
            f"que.*son.*{term_norm}",
            f"definicion.*{term_norm}",
            f"significado.*{term_norm}",
            f"cual.*es.*{term_norm}",
            f"^{term_norm}$"
        ]

    DIRECT_PATTERNS = compile_patterns(DIRECT_PATTERNS_RAW)
    COMPLEX_PATTERNS = compile_list(COMPLEX_PATTERNS_RAW)

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normaliza el texto para mejorar coincidencias del clasificador."""
        text = text.lower()
        text = "".join(c for c in unicodedata.normalize("NFD", text)
                       if unicodedata.category(c) != "Mn")
        return text

    @staticmethod
    def _buscar_definicion_similar(question: str) -> Optional[str]:
        """Busca una definición que coincida con la pregunta usando similitud de texto."""
        q_norm = QuestionClassifier.normalize_text(question)
        
        palabras_ignorar = {'que', 'es', 'la', 'el', 'un', 'una', 'son', 'los', 'las',
                           'definicion', 'de', 'significa', 'significado', 'cual', 'cuales', 'me', 'puedes', 'explicar'}
        palabras_pregunta = [p for p in q_norm.split() if p not in palabras_ignorar and len(p) > 1]
        
        q_limpia = ' '.join(palabras_pregunta)
        
        if not palabras_pregunta:
            return None
        
        mejor_coincidencia = None
        mejor_score = 0
        
        for term_original in DEFINICIONES.keys():
            if len(term_original) > 50: 
                continue
                
            term_norm = QuestionClassifier.normalize_text(term_original)
            
            score = 0
            
            if term_norm in q_norm:
                score = 100
            elif term_norm in q_limpia:
                score = 95
            elif all(palabra in q_norm for palabra in term_norm.split() if len(palabra) > 1):
                score = 85
            else:
                palabras_term = [p for p in term_norm.split() if len(p) > 1]
                if palabras_term:
                    coincidencias = sum(1 for p in palabras_term if p in palabras_pregunta or p in q_norm)
                    porcentaje = coincidencias / len(palabras_term)
                    if porcentaje >= 0.6: 
                        score = porcentaje * 70
            
            if score > mejor_score and score >= 60: 
                mejor_score = score
                mejor_coincidencia = term_original
        
        if mejor_coincidencia:
            return f"definicion_{mejor_coincidencia}"
        return None

    @staticmethod
    def classify(question: str) -> Tuple[str, Optional[str]]:
        """Clasifica la pregunta como ('direct', subtipo) o ('complex', None)."""

        q = QuestionClassifier.normalize_text(question)

        indicadores_definicion = ['que es', 'que son', 'definicion', 'significa', 'significado', 
                                 'cual es', 'cuales son', 'explica']
        if any(ind in q for ind in indicadores_definicion):
            definicion_match = QuestionClassifier._buscar_definicion_similar(question)
            if definicion_match:
                return ("direct", definicion_match)

        for subtipo, patterns in QuestionClassifier.DIRECT_PATTERNS.items():
            if any(p.search(q) for p in patterns):
                return ("direct", subtipo)
        for pattern in QuestionClassifier.COMPLEX_PATTERNS:
            if pattern.search(q):
                return ("complex", None)

        return ("complex", None)


class DirectAnswerBuilder:

    @classmethod
    def build_answer(cls, subtipo: str, datos: Dict) -> str:
        if subtipo not in _ANSWER_BUILDERS:
            return "No tengo información disponible para responder tu pregunta."

        try:
            return _ANSWER_BUILDERS[subtipo](datos)
        except Exception as e:
            return f"Ha ocurrido un error procesando la información: {str(e)}"

    @staticmethod
    def register(name: str):
        def decorator(func):
            _ANSWER_BUILDERS[name] = func
            return func
        return decorator

    @register("horario")
    def _horario(datos):
        materias = datos.get("materias_inscritas_texto", "")
        if not materias:
            return "No cuentas con materias inscritas este período."
        return f"Tu horario de clases es el siguiente:\n{materias}"

    @register("materias_inscritas")
    def _materias_inscritas(datos):
        total = datos.get("total_materias_inscritas", 0)
        mat = datos.get("materias_inscritas_texto", "")
        if total == 0:
            return "No tienes materias inscritas actualmente."
        return f"Estás inscrito en {total} materias:\n{mat}"


    @register("promedio")
    def _promedio(datos):
        prom = datos.get("promedio")
        if prom is None:
            return "No se tiene registrado un promedio en tu expediente."
        return f"Tu promedio general actual es: {prom}"

    @register("creditos")
    def _creditos(datos):
        disp = datos.get("creditos_disponibles", 0)
        return f"Créditos disponibles: {disp}"

    @register("estado")
    def _estado(datos):
        e = datos.get("estado_academico", "No disponible")
        k = datos.get("situacion_kardex", "")
        if k:
            return f"Tu estado académico es: {e}\nSituación en kardex: {k}"
        return f"Tu estado académico es: {e}"

    @register("materias_aprobadas")
    def _aprobadas(datos):
        total = datos.get("total_materias_aprobadas", 0)
        if total == 0:
            return "No tienes materias aprobadas registradas."
        mats = datos.get("materias_aprobadas_texto", "")
        return f"Has aprobado {total} materias:\n{mats}"


    @register("carrera")
    def _carrera(datos):
        return f"Tu carrera es: {datos.get('carrera', 'No disponible')}"

    @register("semestre")
    def _semestre(datos):
        sem = datos.get("semestre_actual")
        if sem is None:
            return "No hay registro de tu semestre actual."
        return f"Actualmente cursas el semestre {sem}."

    @register("datos_personales")
    def _datos(datos):
        return (
            "Datos personales registrados:\n"
            f"- Boleta: {datos.get('boleta','No disponible')}\n"
            f"- Nombre: {datos.get('nombre','No disponible')}\n"
            f"- Correo: {datos.get('correo','No disponible')}\n"
            f"- Teléfono: {datos.get('telefono','No disponible')}\n"
            f"- Dirección: {datos.get('direccion_completa','No disponible')}"
        )

    @register("inscripcion_info")
    def _inscripcion(datos):
        activa = datos.get("reinscripcion_activa", False)
        fecha = datos.get("inscripcion_caduca", "No disponible")
        if activa:
            return f"La reinscripción está activa. Fecha límite: {fecha}"
        return f"La reinscripción no está activa. Última fecha registrada: {fecha}"

    @register("creditos_detalle")
    def _detalle(datos):
        return (
            "Información de créditos:\n"
            f"- Disponibles: {datos.get('creditos_disponibles',0)}"
        )

    @register("programa_info")
    def _programa(datos):
        return (
            "Información académica del programa:\n"
            f"- Carrera: {datos.get('carrera','No disponible')}\n"
            f"- Semestres restantes: {datos.get('semestres_restantes','N/A')}"
        )

    @register("conteo_materias")
    def _conteo(datos):
        return (
            "Resumen de materias:\n"
            f"- Cursando actualmente: {datos.get('total_materias_inscritas',0)}\n"
            f"- Aprobadas: {datos.get('total_materias_aprobadas',0)}"
        )

    @register("kardex_info")
    def _kardex(datos):
        return (
            "Información del kardex:\n"
            f"- Situación: {datos.get('situacion_kardex','No disponible')}\n"
            f"- Promedio general: {datos.get('promedio','No disponible')}\n"
            f"- Materias aprobadas: {datos.get('total_materias_aprobadas',0)}"
        )

    @register("turno_info")
    def _turno(datos):
        # Note: turno_principal doesn't exist in db_utils, would need to extract from materias_inscritas_raw
        return "La información de turno no está disponible actualmente."

    @register("profesores_info")
    def _profesores(datos):
        # Note: profesores list doesn't exist in db_utils, info is in materias_inscritas_texto
        mat_texto = datos.get("materias_inscritas_texto", "")
        if not mat_texto:
            return "No tienes profesores registrados actualmente."
        return f"Información de profesores incluida en tu horario:\n{mat_texto}"

    @register("fechas_semestre")
    def _fechas_semestre(datos):
        f = datos.get("fechas_semestre", {})
        return (
            "Fechas del semestre:\n"
            f"- Inicio: {f.get('inicio_semestre','N/A')}\n"
            f"- Fin: {f.get('fin_semestre','N/A')}\n"
            f"- Período: {f.get('periodo','N/A')}"
        )

    @register("fechas_parciales")
    def _fechas_parciales(datos):
        f = datos.get("fechas_semestre", {})
        return (
            "Fechas de parciales:\n"
            f"- Primer parcial: {f.get('registro_primer_parcial','N/A')} - {f.get('fin_registro_primer_parcial','N/A')}\n"
            f"- Segundo parcial: {f.get('registro_segundo_parcial','N/A')} - {f.get('fin_registro_segundo_parcial','N/A')}\n"
            f"- Tercer parcial: {f.get('registro_tercer_parcial','N/A')} - {f.get('fin_registro_tercer_parcial','N/A')}"
        )

    @register("fechas_ets")
    def _fechas_ets(datos):
        f = datos.get("fechas_semestre", {})
        return (
            "Fechas relacionadas con ETS:\n"
            f"- Evaluación de profesores: {f.get('evalu_profe','N/A')}\n"
            f"- Subida de documentos: {f.get('subir_doc_ets','N/A')} - {f.get('fin_subir_doc_ets','N/A')}\n"
            f"- Evaluación ETS: {f.get('eval_ets','N/A')} - {f.get('fin_evalu_ets','N/A')}\n"
            f"- Calificación ETS: {f.get('cal_ets','N/A')}"
        )

    @register("profesor_grupos")
    def _prof_grupos(datos):
        g = datos.get("grupos_texto", "")
        if not g:
            return "No tienes grupos asignados para este semestre."
        return f"Grupos asignados:\n{g}"

    @register("profesor_calificacion")
    def _prof_calif(datos):
        return (
            f"Tu calificación promedio es {datos.get('calificacion_promedio',0):.1f}, "
            f"con un total de {datos.get('total_resenas',0)} reseñas registradas."
        )

    @register("profesor_resenas")
    def _prof_resenas(datos):
        c = datos.get("ultimos_comentarios", "")
        if not c:
            return "No se han registrado comentarios recientes."
        return f"Últimos comentarios recibidos:\n{c}"

    @register("profesor_fechas")
    def _prof_fechas(datos):
        f = datos.get("fechas_semestre", {})
        return (
            "Fechas relevantes del semestre:\n"
            f"- Inicio: {f.get('inicio_semestre','N/A')}\n"
            f"- Fin: {f.get('fin_semestre','N/A')}\n"
            f"- Evaluación docente: {f.get('evalu_profe','N/A')}\n"
            f"- Registro de calificaciones (1er parcial): {f.get('registro_primer_parcial','N/A')} - {f.get('fin_registro_primer_parcial','N/A')}"
        )

# Registrar builders de definiciones dinámicamente
for term, definition in DEFINICIONES.items():
    key = f"definicion_{term}"
    # Usar default arguments para capturar correctamente las variables del loop
    def make_builder(term_name=term, def_text=definition):
        return lambda datos: f"{term_name}: {def_text}"
    
    _ANSWER_BUILDERS[key] = make_builder()