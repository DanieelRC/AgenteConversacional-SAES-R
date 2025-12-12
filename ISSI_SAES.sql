DROP DATABASE IF EXISTS SAES;

CREATE DATABASE SAES;

USE SAES;

-- ==================================================================
-- 1. DEFINICIÓN DE TABLAS (DDL)
-- ==================================================================

DROP TABLE IF EXISTS fechas_relevantes;

DROP TABLE IF EXISTS avisos;

DROP TABLE IF EXISTS ets;

DROP TABLE IF EXISTS ets_grupo;

DROP TABLE IF EXISTS materia_reprobada;

DROP TABLE IF EXISTS contador;

DROP TABLE IF EXISTS mensaje_chat;

DROP TABLE IF EXISTS lista;

DROP TABLE IF EXISTS borrador_horario;

DROP TABLE IF EXISTS ua_aprobada;

DROP TABLE IF EXISTS kardex;

DROP TABLE IF EXISTS resena;

DROP TABLE IF EXISTS inscripcion;

DROP TABLE IF EXISTS mat_inscritos;

DROP TABLE IF EXISTS horario;

DROP TABLE IF EXISTS distribucion;

DROP TABLE IF EXISTS grupo;

DROP TABLE IF EXISTS unidad_de_aprendizaje;

DROP TABLE IF EXISTS carrera;

DROP TABLE IF EXISTS estudiante;

DROP TABLE IF EXISTS enfermedades;

DROP TABLE IF EXISTS datos_medicos;

DROP TABLE IF EXISTS datos_personales;

CREATE TABLE datos_personales (
    id VARCHAR(15) NOT NULL,
    contrasena TEXT NOT NULL,
    tipo_usuario TEXT NOT NULL,
    nombre TEXT NOT NULL,
    ape_paterno TEXT NOT NULL,
    ape_materno TEXT NOT NULL,
    fecha_nacimiento DATE NOT NULL,
    RFC TEXT,
    tipo_sangre TEXT NOT NULL,
    CURP TEXT NOT NULL,
    nacionalidad TEXT NOT NULL,
    calle TEXT NOT NULL,
    num_exterior TEXT NOT NULL,
    num_interior TEXT NOT NULL,
    codigo_postal TEXT NOT NULL,
    colonia TEXT NOT NULL,
    delegacion TEXT NOT NULL,
    ciudad TEXT NOT NULL,
    telefono TEXT NOT NULL,
    email TEXT NOT NULL,
    foto BLOB,
    grado TEXT,
    carrera TEXT,
    situacion TEXT,
    calificacion FLOAT,
    primera_vez INT,
    CONSTRAINT PK_USUARIOS PRIMARY KEY (id)
);

CREATE TABLE datos_medicos (
    id VARCHAR(15) NOT NULL,
    id_usuario VARCHAR(15) NOT NULL,
    peso FLOAT NOT NULL,
    altura FLOAT NOT NULL,
    tipo_sangre TEXT NOT NULL,
    nss TEXT NOT NULL,
    CONSTRAINT PK_DM PRIMARY KEY (id),
    CONSTRAINT FK_DM_DP FOREIGN KEY (id_usuario) REFERENCES datos_personales (id)
);

CREATE TABLE enfermedades (
    id VARCHAR(15) NOT NULL,
    id_dat_med VARCHAR(15) NOT NULL,
    descri TEXT NOT NULL,
    CONSTRAINT PK_E PRIMARY KEY (id),
    CONSTRAINT FK_E_DM FOREIGN KEY (id_dat_med) REFERENCES datos_medicos (id)
);

CREATE TABLE estudiante (
    id VARCHAR(15) NOT NULL,
    id_usuario VARCHAR(15) NOT NULL,
    promedio FLOAT NOT NULL,
    creditos_disponibles FLOAT NOT NULL,
    estado_academico TEXT NOT NULL,
    CONSTRAINT PK_ES PRIMARY KEY (id),
    CONSTRAINT FK_ES_DP FOREIGN KEY (id_usuario) REFERENCES datos_personales (id)
);

CREATE TABLE carrera (
    nombre VARCHAR(40) NOT NULL,
    creditos_iniciales INT NOT NULL,
    prefijo_grupo TEXT NOT NULL,
    duracion_max INT NOT NULL,
    CONSTRAINT PK_CAR PRIMARY KEY (nombre)
);

CREATE TABLE unidad_de_aprendizaje (
    id VARCHAR(15) NOT NULL,
    nombre TEXT NOT NULL,
    credito FLOAT NOT NULL,
    carrera VARCHAR(40),
    semestre INT NOT NULL,
    tipo TEXT NOT NULL,
    CONSTRAINT PK_UA PRIMARY KEY (id),
    CONSTRAINT FK_UA_CAR FOREIGN KEY (carrera) REFERENCES carrera (nombre)
);

CREATE TABLE grupo (
    id VARCHAR(15) NOT NULL,
    nombre TEXT NOT NULL,
    id_ua VARCHAR(15) NOT NULL,
    id_prof VARCHAR(15) NOT NULL,
    turno TEXT NOT NULL,
    cupo INT NOT NULL,
    salon TEXT,
    reg_final INT,
    reg_extra INT,
    CONSTRAINT PK_GRU PRIMARY KEY (id),
    CONSTRAINT FK_GRU_DP FOREIGN KEY (id_prof) REFERENCES datos_personales (id),
    CONSTRAINT FK_GRU_UA FOREIGN KEY (id_ua) REFERENCES unidad_de_aprendizaje (id)
);

CREATE TABLE distribucion (
    id VARCHAR(15) NOT NULL,
    id_grupo VARCHAR(15) NOT NULL,
    hora_ini TEXT NOT NULL,
    hora_fin TEXT NOT NULL,
    dia TEXT NOT NULL,
    CONSTRAINT PK_DIS PRIMARY KEY (id),
    CONSTRAINT FK_DIS_GRU FOREIGN KEY (id_grupo) REFERENCES grupo (id)
);

CREATE TABLE horario (
    id VARCHAR(15) NOT NULL,
    id_alumno VARCHAR(15) NOT NULL,
    CONSTRAINT PK_HOR PRIMARY KEY (id),
    CONSTRAINT FK_HOR_DP FOREIGN KEY (id_alumno) REFERENCES datos_personales (id)
);

CREATE TABLE mat_inscritos (
    id VARCHAR(15) NOT NULL,
    id_horario VARCHAR(15) NOT NULL,
    id_grupo VARCHAR(15) NOT NULL,
    calificacion_primer FLOAT,
    calificacion_segundo FLOAT,
    calificacion_tercer FLOAT,
    calificacion_final FLOAT,
    extra FLOAT,
    CONSTRAINT PK_MAT PRIMARY KEY (id),
    CONSTRAINT FK_MAT_HOR FOREIGN KEY (id_horario) REFERENCES horario (id),
    CONSTRAINT FK_MAT_GRU FOREIGN KEY (id_grupo) REFERENCES grupo (id)
);

CREATE TABLE inscripcion (
    id VARCHAR(15) NOT NULL,
    id_alumno VARCHAR(15) NOT NULL,
    fecha_hora_in TIMESTAMP NOT NULL,
    fecha_hora_cad TIMESTAMP NOT NULL,
    CONSTRAINT PK_INS PRIMARY KEY (id),
    CONSTRAINT FK_INS_DP FOREIGN KEY (id_alumno) REFERENCES datos_personales (id)
);

CREATE TABLE resena (
    id VARCHAR(15) NOT NULL,
    id_profesor VARCHAR(15) NOT NULL,
    id_alumno VARCHAR(15) NOT NULL,
    calificacion FLOAT NOT NULL,
    comentarios TEXT,
    fecha DATE NOT NULL,
    periodo TEXT,
    CONSTRAINT PK_RE PRIMARY KEY (id),
    CONSTRAINT FK_RE_PRO FOREIGN KEY (id_profesor) REFERENCES datos_personales (id),
    CONSTRAINT FK_RE_ALU FOREIGN KEY (id_alumno) REFERENCES datos_personales (id)
);

CREATE TABLE kardex (
    id VARCHAR(15) NOT NULL,
    id_alumno VARCHAR(15) NOT NULL,
    promedio FLOAT NOT NULL,
    situacion_academica TEXT NOT NULL,
    semestres_restantes INT NOT NULL,
    creditos_obtenidos FLOAT NOT NULL,
    CONSTRAINT PK_KAR PRIMARY KEY (id),
    CONSTRAINT FK_KAR_DP FOREIGN KEY (id_alumno) REFERENCES datos_personales (id)
);

CREATE TABLE ua_aprobada (
    id VARCHAR(15) NOT NULL,
    id_kardex VARCHAR(15) NOT NULL,
    unidad_aprendizaje TEXT NOT NULL,
    calificacion_final FLOAT NOT NULL,
    semestre INT NOT NULL,
    periodo TEXT NOT NULL,
    fecha DATE NOT NULL,
    metodo_aprobado TEXT NOT NULL,
    CONSTRAINT PK_UAA PRIMARY KEY (id),
    CONSTRAINT FK_UAA_KAR FOREIGN KEY (id_kardex) REFERENCES kardex (id)
);

CREATE TABLE borrador_horario (
    id VARCHAR(15) NOT NULL,
    id_grupo VARCHAR(15) NOT NULL,
    id_alumno VARCHAR(15) NOT NULL,
    id_profesor VARCHAR(15) NOT NULL,
    calificacion TEXT NOT NULL,
    materia TEXT NOT NULL,
    horas_lun TEXT,
    horas_mar TEXT,
    horas_mie TEXT,
    horas_jue TEXT,
    horas_vie TEXT,
    creditos_necesarios FLOAT NOT NULL,
    valido TINYINT(1),
    CONSTRAINT PK_BOR PRIMARY KEY (id),
    CONSTRAINT FK_BOR_DP FOREIGN KEY (id_alumno) REFERENCES datos_personales (id),
    CONSTRAINT FK_BOR_PRO FOREIGN KEY (id_profesor) REFERENCES datos_personales (id),
    CONSTRAINT FK_BOR_GRU FOREIGN KEY (id_grupo) REFERENCES grupo (id)
);

CREATE TABLE lista (
    id VARCHAR(15) NOT NULL,
    id_inscrito VARCHAR(15) NOT NULL,
    fecha DATE NOT NULL,
    asistencia TEXT NOT NULL,
    CONSTRAINT PK_LIS PRIMARY KEY (id),
    CONSTRAINT FK_LIS_MAI FOREIGN KEY (id_inscrito) REFERENCES mat_inscritos (id)
);

CREATE TABLE mensaje_chat (
    id VARCHAR(15) NOT NULL,
    id_usuario VARCHAR(15) NOT NULL,
    fecha DATETIME NOT NULL,
    pregunta_realizada TEXT NOT NULL,
    respuesta_obtenida TEXT,
    CONSTRAINT PK_MEN PRIMARY KEY (id),
    CONSTRAINT FK_MEN_DP FOREIGN KEY (id_usuario) REFERENCES datos_personales (id)
);

CREATE TABLE contador (
    id_profesor VARCHAR(15) NOT NULL,
    suma INTEGER NOT NULL,
    registrados INTEGER NOT NULL,
    CONSTRAINT FK_CON_DP FOREIGN KEY (id_profesor) REFERENCES datos_personales (id)
);

create table materia_reprobada (
    id VARCHAR(15) NOT NULL,
    id_estudiante VARCHAR(15) NOT NULL,
    id_ua VARCHAR(15) NOT NULL,
    periodos_restantes integer not null,
    recurse integer not null,
    estado_actual TEXT NOT NULL,
    CONSTRAINT PK_MR PRIMARY KEY (id),
    CONSTRAINT FK_MR_ES FOREIGN KEY (id_estudiante) REFERENCES estudiante (id),
    CONSTRAINT FK_MR_UA FOREIGN KEY (id_ua) REFERENCES unidad_de_aprendizaje (id)
);

create table ets_grupo (
    id VARCHAR(15) NOT NULL,
    id_ua VARCHAR(15) NOT NULL,
    id_aplicante VARCHAR(15) NOT NULL,
    turno TEXT NOT NULL,
    hora_inicio TEXT NOT NULL,
    hora_final TEXT NOT NULL,
    fecha DATE NOT NULL,
    periodo TEXT NOT NULL,
    CONSTRAINT PK_ETS_G PRIMARY KEY (id),
    CONSTRAINT FK_ETSG_UA FOREIGN KEY (id_ua) REFERENCES unidad_de_aprendizaje (id),
    CONSTRAINT FK_ETSG_DP FOREIGN KEY (id_aplicante) REFERENCES datos_personales (id)
);

create table ets (
    id VARCHAR(15) NOT NULL,
    id_mr VARCHAR(15) NOT NULL,
    id_grupo VARCHAR(15) NOT NULL,
    comprobante LONGBLOB,
    validado INTEGER NOT NULL,
    calificado FLOAT NOT NULL,
    CONSTRAINT PK_ETS PRIMARY KEY (id),

-- CASCADE para borrar ETS cuando se elimine la materia reprobada
CONSTRAINT FK_ETS_MR FOREIGN KEY (id_mr) REFERENCES materia_reprobada (id) ON DELETE CASCADE,

-- CASCADE para borrar ETS cuando se elimine el grupo del ETS
CONSTRAINT FK_ETS_AP 
        FOREIGN KEY (id_grupo) 
        REFERENCES ets_grupo (id)
        ON DELETE CASCADE
);

CREATE TABLE avisos (
    id VARCHAR(15) NOT NULL,
    titulo TEXT NOT NULL,
    descripcion TEXT NOT NULL,
    imagen LONGBLOB,
    objetivo TEXT NOT NULL,
    fecha_vencimiento DATETIME NOT NULL,
    CONSTRAINT PK_AVI PRIMARY KEY (id)
);

CREATE TABLE fechas_relevantes (
    inicio_semestre DATETIME NOT NULL,
    fin_semestre DATETIME NOT NULL,
    registro_primer_parcial DATETIME NOT NULL,
    fin_registro_primer_parcial DATETIME NOT NULL,
    registro_segundo_parcial DATETIME NOT NULL,
    fin_registro_segundo_parcial DATETIME NOT NULL,
    registro_tercer_parcial DATETIME NOT NULL,
    fin_registro_tercer_parcial DATETIME NOT NULL,
    registro_extra DATETIME NOT NULL,
    fin_registro_extra DATETIME NOT NULL,
    evalu_profe DATETIME NOT NULL,
    fin_evalu_profe DATETIME NOT NULL,
    inscribir_ets DATETIME NOT NULL,
    fin_inscribir_ets DATETIME NOT NULL,
    subir_doc_ets DATETIME NOT NULL,
    fin_subir_doc_ets DATETIME NOT NULL,
    eval_ets DATETIME NOT NULL,
    fin_evalu_ets DATETIME NOT NULL,
    periodo TEXT NOT NULL
);

-- ==================================================================
-- 2. INSERCIÓN DE DATOS MAESTROS (CARRERAS Y USUARIOS)
-- ==================================================================

INSERT INTO
    carrera (
        nombre,
        creditos_iniciales,
        prefijo_grupo,
        duracion_max
    )
VALUES (
        'Ingenieria en Sistemas Computacionales',
        25,
        'C',
        12
    ),
    (
        'Ingenieria en Inteligencia Artificial',
        25,
        'B',
        12
    ),
    (
        'Licenciatura en Ciencia de Datos',
        25,
        'A',
        12
    );

INSERT INTO
    datos_personales (
        id,
        contrasena,
        tipo_usuario,
        nombre,
        ape_paterno,
        ape_materno,
        fecha_nacimiento,
        RFC,
        tipo_sangre,
        CURP,
        nacionalidad,
        calle,
        num_exterior,
        num_interior,
        codigo_postal,
        colonia,
        delegacion,
        ciudad,
        telefono,
        email,
        foto,
        grado,
        carrera,
        situacion,
        calificacion
    )
VALUES (
        '2023635321',
        '$2b$10$FVM7xU.TbCeLwfOmDq1QeeF0DCu6FlpuS8f.hqSL55HmZojG1ZbNG',
        'administrador',
        'Juan',
        'Pérez',
        'González',
        '2007-05-23',
        'ABCDEFGE',
        'O+',
        'ABCDEFG',
        'mexicana',
        'Héroes',
        '15',
        'n/a',
        '12345',
        'Juárez',
        'Tlalpan',
        'CDMX',
        '123456',
        'juan_perez@gmail.com',
        NULL,
        NULL,
        NULL,
        'activo',
        NULL
    ),
    (
        'HIJKLMNO',
        '$2b$10$FVM7xU.TbCeLwfOmDq1QeeF0DCu6FlpuS8f.hqSL55HmZojG1ZbNG',
        'profesor',
        'María',
        'López',
        'Martínez',
        '1980-11-15',
        'HIJKLMNO',
        'A+',
        'HIJKLMNO',
        'mexicana',
        'Revolución',
        '45',
        'n/a',
        '67890',
        'Centro',
        'Coyoacán',
        'CDMX',
        '654321',
        'maria_lopez@gmail.com',
        NULL,
        'Doctorado en Ciencias',
        NULL,
        'activo',
        9.0
    ),
    (
        'P0001RSV',
        '$2b$10$FVM7xU.TbCeLwfOmDq1QeeF0DCu6FlpuS8f.hqSL55HmZojG1ZbNG',
        'profesor',
        'Ricardo',
        'Solís',
        'Vega',
        '1975-08-10',
        'RSV750810ABC',
        'O-',
        'RSV750810HDFLPR03',
        'mexicana',
        'Calle 10',
        '101',
        'B',
        '03100',
        'Narvarte',
        'Benito Juárez',
        'CDMX',
        '5598765432',
        'r.solis.v@escom.ipn.mx',
        NULL,
        'Maestría en TI',
        NULL,
        'activo',
        8.5
    ),
    (
        'P0002KND',
        '$2b$10$FVM7xU.TbCeLwfOmDq1QeeF0DCu6FlpuS8f.hqSL55HmZojG1ZbNG',
        'profesor',
        'Karla',
        'Núñez',
        'Díaz',
        '1990-03-05',
        'KND900305ABC',
        'B+',
        'KND900305HDFLPR03',
        'mexicana',
        'Av. Cien',
        '25',
        '3A',
        '07500',
        'San Juan',
        'GAM',
        'CDMX',
        '5511223344',
        'karla.nunez@escom.ipn.mx',
        NULL,
        'Doctorado en IA',
        NULL,
        'activo',
        9.3
    ),
    (
        'P0003HGR',
        '$2b$10$FVM7xU.TbCeLwfOmDq1QeeF0DCu6FlpuS8f.hqSL55HmZojG1ZbNG',
        'profesor',
        'Héctor',
        'Gil',
        'Ramos',
        '1985-12-24',
        'HGR851224ABC',
        'A-',
        'HGR851224HDFLPR03',
        'mexicana',
        'Eje Central',
        '50',
        '10B',
        '06000',
        'Centro Histórico',
        'Cuauhtémoc',
        'CDMX',
        '5544332211',
        'hector.gil@escom.ipn.mx',
        NULL,
        'Maestría en Ciencia de Datos',
        NULL,
        'activo',
        8.9
    );

-- ==================================================================
-- 3. INSERCIÓN DE UNIDADES DE APRENDIZAJE (MANDATORIAS Y OPTATIVAS)
-- ==================================================================
INSERT INTO
    unidad_de_aprendizaje (
        id,
        nombre,
        credito,
        carrera,
        semestre,
        tipo
    )
VALUES
    -- ISC (Ingenieria en Sistemas Computacionales) - OBLIGATORIAS
    (
        'UA0001',
        'Fundamentos de Programación',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        1,
        'OBLIGATORIA'
    ),
    (
        'UA0002',
        'Matemáticas Discretas',
        10.5,
        'Ingenieria en Sistemas Computacionales',
        1,
        'OBLIGATORIA'
    ),
    (
        'UA0003',
        'Cálculo',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        1,
        'OBLIGATORIA'
    ),
    (
        'UA0004',
        'Análisis Vectorial',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        1,
        'OBLIGATORIA'
    ),
    (
        'UA0005',
        'Comunicación Oral y Escrita',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        1,
        'OBLIGATORIA'
    ),
    (
        'UA0006',
        'Álgebra Lineal',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        2,
        'OBLIGATORIA'
    ),
    (
        'UA0007',
        'Cálculo Aplicado',
        9.0,
        'Ingenieria en Sistemas Computacionales',
        2,
        'OBLIGATORIA'
    ),
    (
        'UA0008',
        'Mecánica y Electromagnetismo',
        10.5,
        'Ingenieria en Sistemas Computacionales',
        2,
        'OBLIGATORIA'
    ),
    (
        'UA0009',
        'Ingeniería, Ética y Sociedad',
        9.0,
        'Ingenieria en Sistemas Computacionales',
        2,
        'OBLIGATORIA'
    ),
    (
        'UA0010',
        'Fundamentos Económicos',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        2,
        'OBLIGATORIA'
    ),
    (
        'UA0011',
        'Algoritmos y Estructuras de Datos',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        2,
        'OBLIGATORIA'
    ),
    (
        'UA0012',
        'Ecuaciones Diferenciales',
        9.0,
        'Ingenieria en Sistemas Computacionales',
        3,
        'OBLIGATORIA'
    ),
    (
        'UA0013',
        'Circuitos Eléctricos',
        10.5,
        'Ingenieria en Sistemas Computacionales',
        3,
        'OBLIGATORIA'
    ),
    (
        'UA0014',
        'Fundamentos de Diseño Digital',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        3,
        'OBLIGATORIA'
    ),
    (
        'UA0015',
        'Bases de Datos',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        3,
        'OBLIGATORIA'
    ),
    (
        'UA0016',
        'Finanzas Empresariales',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        3,
        'OBLIGATORIA'
    ),
    (
        'UA0017',
        'Paradigmas de Programación',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        3,
        'OBLIGATORIA'
    ),
    (
        'UA0018',
        'Análisis y Diseño de Algoritmos',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        3,
        'OBLIGATORIA'
    ),
    (
        'UA0019',
        'Probabilidad y Estadística',
        9.0,
        'Ingenieria en Sistemas Computacionales',
        4,
        'OBLIGATORIA'
    ),
    (
        'UA0020',
        'Matemáticas Avanzadas para la Ingeniería',
        9.0,
        'Ingenieria en Sistemas Computacionales',
        4,
        'OBLIGATORIA'
    ),
    (
        'UA0021',
        'Electrónica Analógica',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        4,
        'OBLIGATORIA'
    ),
    (
        'UA0022',
        'Diseño de Sistemas Digitales',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        4,
        'OBLIGATORIA'
    ),
    (
        'UA0023',
        'Tecnologías para el Desarrollo de Aplicaciones Web',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        4,
        'OBLIGATORIA'
    ),
    (
        'UA0024',
        'Sistemas Operativos',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        4,
        'OBLIGATORIA'
    ),
    (
        'UA0025',
        'Teoría de la Computación',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        4,
        'OBLIGATORIA'
    ),
    (
        'UA0026',
        'Procesamiento Digital de Señales',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        5,
        'OBLIGATORIA'
    ),
    (
        'UA0027',
        'Instrumentación y Control',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        5,
        'OBLIGATORIA'
    ),
    (
        'UA0028',
        'Arquitectura de Computadoras',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        5,
        'OBLIGATORIA'
    ),
    (
        'UA0029',
        'Análisis y Diseño de Sistemas',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        5,
        'OBLIGATORIA'
    ),
    (
        'UA0030',
        'Formulación y Evaluación de Proyectos Informáticos',
        6.0,
        'Ingenieria en Sistemas Computacionales',
        5,
        'OBLIGATORIA'
    ),
    (
        'UA0031',
        'Compiladores',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        5,
        'OBLIGATORIA'
    ),
    (
        'UA0032',
        'Redes de Computadoras',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        5,
        'OBLIGATORIA'
    ),
    (
        'UA0033',
        'Sistemas en Chip',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        6,
        'OBLIGATORIA'
    ),
    (
        'UA0036',
        'Métodos Cuantitativos para la Toma de Decisiones',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        6,
        'OBLIGATORIA'
    ),
    (
        'UA0037',
        'Ingeniería de Software',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        6,
        'OBLIGATORIA'
    ),
    (
        'UA0038',
        'Inteligencia Artificial',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        6,
        'OBLIGATORIA'
    ),
    (
        'UA0039',
        'Aplicaciones para Comunicaciones en Red',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        6,
        'OBLIGATORIA'
    ),
    (
        'UA0040',
        'Desarrollo de Aplicaciones Móviles Nativas',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        7,
        'OBLIGATORIA'
    ),
    (
        'UA0043',
        'Sistemas Distribuidos',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        7,
        'OBLIGATORIA'
    ),
    (
        'UA0044',
        'Administración de Servicios en Red',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        7,
        'OBLIGATORIA'
    ),
    (
        'UA0045',
        'Estancia Profesional',
        12.0,
        'Ingenieria en Sistemas Computacionales',
        8,
        'OBLIGATORIA'
    ),
    (
        'UA0046',
        'Desarrollo de Habilidades Sociales para la Alta Dirección',
        3.0,
        'Ingenieria en Sistemas Computacionales',
        8,
        'OBLIGATORIA'
    ),
    (
        'UA0047',
        'Trabajo Terminal II',
        12.0,
        'Ingenieria en Sistemas Computacionales',
        8,
        'OBLIGATORIA'
    ),
    (
        'UA0048',
        'Gestión Empresarial',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        8,
        'OBLIGATORIA'
    ),
    (
        'UA0049',
        'Liderazgo Personal',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        8,
        'OBLIGATORIA'
    ),
    (
        'UA0050',
        'Trabajo Terminal I',
        12.0,
        'Ingenieria en Sistemas Computacionales',
        8,
        'OBLIGATORIA'
    ),
    -- IIA (Ingenieria en Inteligencia Artificial) - OBLIGATORIAS
    (
        'UA0051',
        'Fundamentos de Programación',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        1,
        'OBLIGATORIA'
    ),
    (
        'UA0052',
        'Matemáticas Discretas',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        1,
        'OBLIGATORIA'
    ),
    (
        'UA0053',
        'Cálculo',
        10.5,
        'Ingenieria en Inteligencia Artificial',
        1,
        'OBLIGATORIA'
    ),
    (
        'UA0054',
        'Comunicación Oral y Escrita',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        1,
        'OBLIGATORIA'
    ),
    (
        'UA0055',
        'Fundamentos Económicos',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        1,
        'OBLIGATORIA'
    ),
    (
        'UA0056',
        'Algoritmos y Estructuras de Datos',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        2,
        'OBLIGATORIA'
    ),
    (
        'UA0057',
        'Álgebra Lineal',
        9.0,
        'Ingenieria en Inteligencia Artificial',
        2,
        'OBLIGATORIA'
    ),
    (
        'UA0058',
        'Mecánica y Electromagnetismo',
        10.5,
        'Ingenieria en Inteligencia Artificial',
        2,
        'OBLIGATORIA'
    ),
    (
        'UA0059',
        'Ingeniería, Ética y Sociedad',
        9.0,
        'Ingenieria en Inteligencia Artificial',
        2,
        'OBLIGATORIA'
    ),
    (
        'UA0060',
        'Liderazgo Personal',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        2,
        'OBLIGATORIA'
    ),
    (
        'UA0061',
        'Análisis y Diseño de Algoritmos',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        3,
        'OBLIGATORIA'
    ),
    (
        'UA0062',
        'Cálculo Aplicado',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        3,
        'OBLIGATORIA'
    ),
    (
        'UA0063',
        'Ecuaciones Diferenciales',
        9.0,
        'Ingenieria en Inteligencia Artificial',
        3,
        'OBLIGATORIA'
    ),
    (
        'UA0064',
        'Fundamentos de Diseño Digital',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        3,
        'OBLIGATORIA'
    ),
    (
        'UA0065',
        'Bases de Datos',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        3,
        'OBLIGATORIA'
    ),
    (
        'UA0066',
        'Finanzas Empresariales',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        3,
        'OBLIGATORIA'
    ),
    (
        'UA0067',
        'Teoría de la Computación',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        4,
        'OBLIGATORIA'
    ),
    (
        'UA0068',
        'Probabilidad y Estadística',
        9.0,
        'Ingenieria en Inteligencia Artificial',
        4,
        'OBLIGATORIA'
    ),
    (
        'UA0069',
        'Matemáticas Avanzadas para la Ingeniería',
        9.0,
        'Ingenieria en Inteligencia Artificial',
        4,
        'OBLIGATORIA'
    ),
    (
        'UA0070',
        'Fundamentos de Inteligencia Artificial',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        4,
        'OBLIGATORIA'
    ),
    (
        'UA0071',
        'Diseño de Sistemas Digitales',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        4,
        'OBLIGATORIA'
    ),
    (
        'UA0072',
        'Análisis y Diseño de Sistemas',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        4,
        'OBLIGATORIA'
    ),
    (
        'UA0073',
        'Procesamiento de Señales',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        5,
        'OBLIGATORIA'
    ),
    (
        'UA0074',
        'Paradigmas de Programación',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        5,
        'OBLIGATORIA'
    ),
    (
        'UA0075',
        'Tecnologías para el Desarrollo de Aplicaciones Web',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        5,
        'OBLIGATORIA'
    ),
    (
        'UA0076',
        'Procesamiento Digital de Señales',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        5,
        'OBLIGATORIA'
    ),
    (
        'UA0077',
        'Gestión Empresarial',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        5,
        'OBLIGATORIA'
    ),
    (
        'UA0078',
        'Aprendizaje de Máquina',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        6,
        'OBLIGATORIA'
    ),
    (
        'UA0079',
        'Tecnologías de Lenguaje Natural',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        6,
        'OBLIGATORIA'
    ),
    (
        'UA0080',
        'Cómputo Paralelo',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        6,
        'OBLIGATORIA'
    ),
    (
        'UA0081',
        'Ingeniería de Software para Sistemas Inteligentes',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        6,
        'OBLIGATORIA'
    ),
    (
        'UA0082',
        'Metodología de la Investigación y Divulgación Científica',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        6,
        'OBLIGATORIA'
    ),
    (
        'UA0083',
        'Visión Artificial',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        7,
        'OBLIGATORIA'
    ),
    (
        'UA0084',
        'Reconocimiento de Voz',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        7,
        'OBLIGATORIA'
    ),
    (
        'UA0085',
        'Algoritmos Bioinspirados',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        7,
        'OBLIGATORIA'
    ),
    (
        'UA0086',
        'Redes Neuronales y Aprendizaje Profundo',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        7,
        'OBLIGATORIA'
    ),
    (
        'UA0089',
        'Formulación y Evaluación de Proyectos Informáticos',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        8,
        'OBLIGATORIA'
    ),
    (
        'UA0090',
        'Trabajo Terminal I',
        12.0,
        'Ingenieria en Inteligencia Artificial',
        8,
        'OBLIGATORIA'
    ),
    (
        'UA0091',
        'Trabajo Terminal II',
        12.0,
        'Ingenieria en Inteligencia Artificial',
        8,
        'OBLIGATORIA'
    ),
    (
        'UA0092',
        'Estancia Profesional',
        12.0,
        'Ingenieria en Inteligencia Artificial',
        8,
        'OBLIGATORIA'
    ),
    (
        'UA0093',
        'Desarrollo de Habilidades Sociales para la Alta Dirección',
        3.0,
        'Ingenieria en Inteligencia Artificial',
        8,
        'OBLIGATORIA'
    ),
    -- LCD (Licenciatura en Ciencia de Datos) - OBLIGATORIAS
    (
        'UA0096',
        'Fundamentos de Programación',
        7.5,
        'Licenciatura en Ciencia de Datos',
        1,
        'OBLIGATORIA'
    ),
    (
        'UA0097',
        'Matemáticas Discretas',
        10.5,
        'Licenciatura en Ciencia de Datos',
        1,
        'OBLIGATORIA'
    ),
    (
        'UA0098',
        'Cálculo',
        7.5,
        'Licenciatura en Ciencia de Datos',
        1,
        'OBLIGATORIA'
    ),
    (
        'UA0099',
        'Comunicación Oral y Escrita',
        7.5,
        'Licenciatura en Ciencia de Datos',
        1,
        'OBLIGATORIA'
    ),
    (
        'UA0100',
        'Introducción a la Ciencia de Datos',
        6.0,
        'Licenciatura en Ciencia de Datos',
        1,
        'OBLIGATORIA'
    ),
    (
        'UA0101',
        'Algoritmos y Estructuras de Datos',
        7.5,
        'Licenciatura en Ciencia de Datos',
        2,
        'OBLIGATORIA'
    ),
    (
        'UA0102',
        'Álgebra Lineal',
        7.5,
        'Licenciatura en Ciencia de Datos',
        2,
        'OBLIGATORIA'
    ),
    (
        'UA0103',
        'Cálculo Multivariable',
        10.5,
        'Licenciatura en Ciencia de Datos',
        2,
        'OBLIGATORIA'
    ),
    (
        'UA0104',
        'Ética y Legalidad',
        9.0,
        'Licenciatura en Ciencia de Datos',
        2,
        'OBLIGATORIA'
    ),
    (
        'UA0105',
        'Fundamentos Económicos',
        7.5,
        'Licenciatura en Ciencia de Datos',
        2,
        'OBLIGATORIA'
    ),
    (
        'UA0106',
        'Análisis y Diseño de Algoritmos',
        7.5,
        'Licenciatura en Ciencia de Datos',
        3,
        'OBLIGATORIA'
    ),
    (
        'UA0107',
        'Programación para Ciencia de Datos',
        7.5,
        'Licenciatura en Ciencia de Datos',
        3,
        'OBLIGATORIA'
    ),
    (
        'UA0108',
        'Probabilidad',
        10.5,
        'Licenciatura en Ciencia de Datos',
        3,
        'OBLIGATORIA'
    ),
    (
        'UA0109',
        'Bases de Datos',
        7.5,
        'Licenciatura en Ciencia de Datos',
        3,
        'OBLIGATORIA'
    ),
    (
        'UA0110',
        'Métodos Numéricos',
        10.5,
        'Licenciatura en Ciencia de Datos',
        3,
        'OBLIGATORIA'
    ),
    (
        'UA0111',
        'Finanzas Empresariales',
        7.5,
        'Licenciatura en Ciencia de Datos',
        3,
        'OBLIGATORIA'
    ),
    (
        'UA0112',
        'Desarrollo de Aplicaciones Web',
        7.5,
        'Licenciatura en Ciencia de Datos',
        4,
        'OBLIGATORIA'
    ),
    (
        'UA0113',
        'Cómputo de Alto Desempeño',
        9.0,
        'Licenciatura en Ciencia de Datos',
        4,
        'OBLIGATORIA'
    ),
    (
        'UA0114',
        'Estadística',
        10.5,
        'Licenciatura en Ciencia de Datos',
        4,
        'OBLIGATORIA'
    ),
    (
        'UA0115',
        'Base de Datos Avanzadas',
        10.5,
        'Licenciatura en Ciencia de Datos',
        4,
        'OBLIGATORIA'
    ),
    (
        'UA0116',
        'Desarrollo de Aplicaciones para Análisis de Datos',
        7.5,
        'Licenciatura en Ciencia de Datos',
        4,
        'OBLIGATORIA'
    ),
    (
        'UA0117',
        'Liderazgo Personal',
        7.5,
        'Licenciatura en Ciencia de Datos',
        4,
        'OBLIGATORIA'
    ),
    (
        'UA0118',
        'Minería de Datos',
        7.5,
        'Licenciatura en Ciencia de Datos',
        5,
        'OBLIGATORIA'
    ),
    (
        'UA0119',
        'Matemáticas Avanzadas para Ciencia de Datos',
        10.5,
        'Licenciatura en Ciencia de Datos',
        5,
        'OBLIGATORIA'
    ),
    (
        'UA0120',
        'Procesos Estocásticos',
        10.5,
        'Licenciatura en Ciencia de Datos',
        5,
        'OBLIGATORIA'
    ),
    (
        'UA0121',
        'Aprendizaje de Máquina e Inteligencia Artificial',
        7.5,
        'Licenciatura en Ciencia de Datos',
        5,
        'OBLIGATORIA'
    ),
    (
        'UA0122',
        'Analítica y Visualización de Datos',
        7.5,
        'Licenciatura en Ciencia de Datos',
        5,
        'OBLIGATORIA'
    ),
    (
        'UA0123',
        'Metodología de la Investigación y Divulgación Científica',
        7.5,
        'Licenciatura en Ciencia de Datos',
        5,
        'OBLIGATORIA'
    ),
    (
        'UA0124',
        'Modelado Predictivo',
        7.5,
        'Licenciatura en Ciencia de Datos',
        6,
        'OBLIGATORIA'
    ),
    (
        'UA0125',
        'Procesamiento de Lenguaje Natural',
        4.5,
        'Licenciatura en Ciencia de Datos',
        6,
        'OBLIGATORIA'
    ),
    (
        'UA0126',
        'Análisis de Series de Tiempo',
        4.5,
        'Licenciatura en Ciencia de Datos',
        6,
        'OBLIGATORIA'
    ),
    (
        'UA0127',
        'Analítica Avanzada de Datos',
        7.5,
        'Licenciatura en Ciencia de Datos',
        6,
        'OBLIGATORIA'
    ),
    (
        'UA0130',
        'Big Data',
        7.5,
        'Licenciatura en Ciencia de Datos',
        7,
        'OBLIGATORIA'
    ),
    (
        'UA0131',
        'Modelos Econométricos',
        4.5,
        'Licenciatura en Ciencia de Datos',
        7,
        'OBLIGATORIA'
    ),
    (
        'UA0132',
        'Trabajo Terminal I',
        12.0,
        'Licenciatura en Ciencia de Datos',
        7,
        'OBLIGATORIA'
    ),
    (
        'UA0133',
        'Administración de Proyectos de TI',
        10.5,
        'Licenciatura en Ciencia de Datos',
        7,
        'OBLIGATORIA'
    ),
    (
        'UA0136',
        'Desarrollo de Habilidades Sociales para la Alta Dirección',
        6.0,
        'Licenciatura en Ciencia de Datos',
        8,
        'OBLIGATORIA'
    ),
    (
        'UA0137',
        'Gestión Empresarial',
        10.5,
        'Licenciatura en Ciencia de Datos',
        8,
        'OBLIGATORIA'
    ),
    (
        'UA0138',
        'Trabajo Terminal II',
        12.0,
        'Licenciatura en Ciencia de Datos',
        8,
        'OBLIGATORIA'
    ),
    (
        'UA0139',
        'Estancia Profesional',
        4.5,
        'Licenciatura en Ciencia de Datos',
        8,
        'OBLIGATORIA'
    ),
    -- OPTATIVAS
    (
        'UA0140',
        'Tópicos Selectos de Criptografía',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        6,
        'OPTATIVA'
    ),
    (
        'UA0141',
        'Introducción a la Criptografía',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        6,
        'OPTATIVA'
    ),
    (
        'UA0142',
        'Seguridad Informática',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        6,
        'OPTATIVA'
    ),
    (
        'UA0143',
        'Aseguramiento de Calidad de Software y Patrones de Diseño',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        6,
        'OPTATIVA'
    ),
    (
        'UA0144',
        'Gestión de Empresas de Alta Tecnología',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        6,
        'OPTATIVA'
    ),
    (
        'UA0145',
        'Herramientas Estadísticas para Análisis de Datos',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        6,
        'OPTATIVA'
    ),
    (
        'UA0146',
        'Algoritmos Genéticos',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        6,
        'OPTATIVA'
    ),
    (
        'UA0147',
        'Bases de Datos No Relacionales',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        6,
        'OPTATIVA'
    ),
    (
        'UA0148',
        'Sistemas Embebidos',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        6,
        'OPTATIVA'
    ),
    (
        'UA0149',
        'Tópicos Selectos de Computación I',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        6,
        'OPTATIVA'
    ),
    (
        'UA0150',
        'Aprendizaje de Máquina (ISC)',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        6,
        'OPTATIVA'
    ),
    (
        'UA0151',
        'Instrumentación Virtual',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        6,
        'OPTATIVA'
    ),
    (
        'UA0152',
        'Gráficos por Computadora',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        6,
        'OPTATIVA'
    ),
    (
        'UA0153',
        'Autómatas Celulares',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        6,
        'OPTATIVA'
    ),
    (
        'UA0154',
        'Análisis de Imagen',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        7,
        'OPTATIVA'
    ),
    (
        'UA0155',
        'Frameworks de Desarrollo Web Cliente y Servidor',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        7,
        'OPTATIVA'
    ),
    (
        'UA0156',
        'Ingeniería Económica',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        7,
        'OPTATIVA'
    ),
    (
        'UA0157',
        'Minería de Datos (ISC)',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        7,
        'OPTATIVA'
    ),
    (
        'UA0158',
        'Internet de las Cosas',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        7,
        'OPTATIVA'
    ),
    (
        'UA0159',
        'Tópicos Selectos de Computación II',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        7,
        'OPTATIVA'
    ),
    (
        'UA0160',
        'Aplicaciones de Instrumentación Virtual',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        7,
        'OPTATIVA'
    ),
    (
        'UA0161',
        'Procesamiento de Lenguaje Natural (ISC)',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        7,
        'OPTATIVA'
    ),
    (
        'UA0162',
        'Realidad Virtual y Aumentada',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        7,
        'OPTATIVA'
    ),
    (
        'UA0163',
        'Sistemas Complejos',
        7.5,
        'Ingenieria en Sistemas Computacionales',
        7,
        'OPTATIVA'
    ),
    (
        'UA0164',
        'Aplicaciones de Lenguaje Natural',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        6,
        'OPTATIVA'
    ),
    (
        'UA0165',
        'Cómputo en la Nube',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        6,
        'OPTATIVA'
    ),
    (
        'UA0166',
        'Interacción Humano-Máquina',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        6,
        'OPTATIVA'
    ),
    (
        'UA0167',
        'Minería de Datos (IIA)',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        6,
        'OPTATIVA'
    ),
    (
        'UA0168',
        'Programación de Dispositivos Móviles',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        6,
        'OPTATIVA'
    ),
    (
        'UA0169',
        'Propiedad Intelectual (IIA)',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        6,
        'OPTATIVA'
    ),
    (
        'UA0170',
        'Sistemas Multiagentes',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        6,
        'OPTATIVA'
    ),
    (
        'UA0171',
        'Temas Selectos de Inteligencia Artificial',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        7,
        'OPTATIVA'
    ),
    (
        'UA0172',
        'Técnicas de Programación para Robots Móviles',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        7,
        'OPTATIVA'
    ),
    (
        'UA0173',
        'Tópicos Selectos de Algoritmos Bioinspirados',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        7,
        'OPTATIVA'
    ),
    (
        'UA0174',
        'Big Data (IIA)',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        7,
        'OPTATIVA'
    ),
    (
        'UA0175',
        'Aplicaciones de Inteligencia Artificial en Sistemas Embebidos',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        7,
        'OPTATIVA'
    ),
    (
        'UA0176',
        'Innovación y Emprendimiento Tecnológico (IIA)',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        7,
        'OPTATIVA'
    ),
    (
        'UA0177',
        'Aplicaciones de Sistemas Multiagentes',
        7.5,
        'Ingenieria en Inteligencia Artificial',
        7,
        'OPTATIVA'
    ),
    (
        'UA0178',
        'Bioinformática Básica',
        7.5,
        'Licenciatura en Ciencia de Datos',
        6,
        'OPTATIVA'
    ),
    (
        'UA0179',
        'Ciberseguridad',
        7.5,
        'Licenciatura en Ciencia de Datos',
        6,
        'OPTATIVA'
    ),
    (
        'UA0180',
        'Estadística Avanzada',
        7.5,
        'Licenciatura en Ciencia de Datos',
        6,
        'OPTATIVA'
    ),
    (
        'UA0181',
        'Propiedad Intelectual (LCD)',
        7.5,
        'Licenciatura en Ciencia de Datos',
        6,
        'OPTATIVA'
    ),
    (
        'UA0182',
        'Simulación Básica',
        7.5,
        'Licenciatura en Ciencia de Datos',
        6,
        'OPTATIVA'
    ),
    (
        'UA0183',
        'Sistemas de Información Geográfica',
        7.5,
        'Licenciatura en Ciencia de Datos',
        6,
        'OPTATIVA'
    ),
    (
        'UA0184',
        'Bioinformática Avanzada',
        7.5,
        'Licenciatura en Ciencia de Datos',
        7,
        'OPTATIVA'
    ),
    (
        'UA0185',
        'Protección de Datos',
        7.5,
        'Licenciatura en Ciencia de Datos',
        7,
        'OPTATIVA'
    ),
    (
        'UA0186',
        'Temas Selectos de Aprendizaje Profundo',
        7.5,
        'Licenciatura en Ciencia de Datos',
        7,
        'OPTATIVA'
    ),
    (
        'UA0187',
        'Innovación y Emprendimiento Tecnológico (LCD)',
        7.5,
        'Licenciatura en Ciencia de Datos',
        7,
        'OPTATIVA'
    ),
    (
        'UA0188',
        'Simulación Avanzada',
        7.5,
        'Licenciatura en Ciencia de Datos',
        7,
        'OPTATIVA'
    ),
    (
        'UA0189',
        'Temas Selectos de Inteligencia Artificial (LCD)',
        7.5,
        'Licenciatura en Ciencia de Datos',
        7,
        'OPTATIVA'
    ),
    (
        'UA0190',
        'Temas Selectos de Procesamiento de Lenguaje Natural',
        7.5,
        'Licenciatura en Ciencia de Datos',
        7,
        'OPTATIVA'
    );

-- ==================================================================
-- 3. INSERCIÓN DE GRUPOS Y DISTRIBUCIÓN DE HORARIOS
-- Se crean grupos Matutinos (M1) y Vespertinos (V1) para todas las UAs OBLIGATORIAS.
-- ==================================================================
DELETE FROM grupo;

DELETE FROM distribucion;

-- Definición de Patrones de Horario (1.5 horas por sesión)
-- Slot 1: 7:00-8:30 (M) | 3:00-4:30 (V)
-- Slot 2: 8:30-10:00 (M)| 5:00-6:30 (V)
-- Slot 3: 10:30-12:00 (M)| 6:30-8:00 (V)
-- Slot 4: 12:00-1:30 (M) | 8:00-9:30 (V)
-- Slot 5: 1:30-3:00 (M) | N/A (V)

SET @grupo_id_counter = 1;

SET @dist_id_counter = 1;

-- Lista de profesores para rotación
SET @profesores = 'HIJKLMNO,P0001RSV,P0002KND,P0003HGR';

SET @num_profesores = 4;

-- Mapeo de slots de tiempo
SET
    @slot_map_M = 'T1=7:00-8:30,T2=8:30-10:00,T3=10:30-12:00,T4=12:00-1:30,T5=1:30-3:00';

SET
    @slot_map_V = 'T1=3:00-4:30,T2=5:00-6:30,T3=6:30-8:00,T4=8:00-9:30';

-- 3 Sesiones (7.5/6.0 créditos)
SET @D3 = 'Lunes,Martes,Miércoles';

SET @S3_0 = 'T1,T2,T3';
-- Horario 1: (L:T1, M:T2, X:T3)
SET @S3_1 = 'T4,T5,T1';
-- Horario 2: (L:T4, M:T5, X:T1)
SET @S3_2 = 'T2,T3,T4';
-- Horario 3: (L:T2, M:T3, X:T4)
SET @S3_3 = 'T5,T1,T2';
-- Horario 4: (L:T5, M:T1, X:T2)
SET @S3_4 = 'T3,T4,T5';
-- Horario 5: (L:T3, M:T4, X:T5)

-- 4 Sesiones (9.0 créditos)
SET @D4 = 'Lunes,Martes,Miércoles,Jueves';

SET @S4_0 = 'T1,T2,T3,T4';
-- Horario 1: (L:T1, M:T2, X:T3, J:T4)
SET @S4_1 = 'T5,T1,T2,T3';
-- Horario 2: (L:T5, M:T1, X:T2, J:T3)
SET @S4_2 = 'T4,T5,T1,T2';
-- Horario 3: (L:T4, M:T5, X:T1, J:T2)
SET @S4_3 = 'T3,T4,T5,T1';
-- Horario 4: (L:T3, M:T4, X:T5, J:T1)

-- 5 Sesiones (10.5 créditos)
SET @D5 = 'Lunes,Martes,Miércoles,Jueves,Viernes';

SET @S5_0 = 'T1,T2,T3,T4,T5';
-- Horario 1: (L:T1, M:T2, X:T3, J:T4, V:T5)
SET @S5_1 = 'T5,T1,T2,T3,T4';
-- Horario 2: (L:T5, M:T1, X:T2, J:T3, V:T4)

-- 1-2 Sesiones (3.0/4.5/12.0 créditos - Materias de baja frecuencia)
SET @D2_LOW = 'Martes,Jueves';

SET @S2_0 = 'T4,T4';
-- (M:T4, J:T4)
SET @S2_1 = 'T5,T5';
-- (M:T5, J:T5)

-- Cursor para iterar sobre todas las UAs Obligatorias
DELIMITER $$

CREATE PROCEDURE generate_groups_and_schedules()
BEGIN
    DECLARE done INT DEFAULT 0;
    DECLARE ua_id_val VARCHAR(15);
    DECLARE ua_carrera VARCHAR(40);
    DECLARE ua_sem INT;
    DECLARE ua_credito FLOAT;
    
    DECLARE prefijo CHAR(1);
    DECLARE sem_name VARCHAR(15);
    DECLARE prof_id VARCHAR(15);
    
    -- Variables de control de la iteración (para rotación de horarios por semestre)
    DECLARE current_sem INT DEFAULT 0;
    DECLARE current_carrera VARCHAR(40) DEFAULT '';
    DECLARE semester_ua_index INT DEFAULT 0; -- Índice para rotación de slots (0, 1, 2, 3...)
    
    -- Definir cursor para UAs OBLIGATORIAS
    DECLARE ua_cursor CURSOR FOR
        SELECT id, carrera, semestre, credito FROM unidad_de_aprendizaje WHERE tipo = 'OBLIGATORIA' ORDER BY carrera, semestre, id;
        
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;

    OPEN ua_cursor;
    
    -- Iteración con LOOP
    ua_loop: LOOP
        FETCH ua_cursor INTO ua_id_val, ua_carrera, ua_sem, ua_credito;
        IF done THEN
            LEAVE ua_loop;
        END IF;

        -- Lógica para restablecer el índice de rotación al inicio de un nuevo bloque Semestre-Carrera
        IF ua_sem != current_sem OR ua_carrera != current_carrera THEN
            SET current_sem = ua_sem;
            SET current_carrera = ua_carrera;
            SET semester_ua_index = 0; -- Resetear el índice de rotación para este semestre
        END IF;
        
        SET prefijo = (SELECT prefijo_grupo FROM carrera WHERE nombre = ua_carrera);
        SET sem_name = CONCAT(ua_sem, prefijo);

        -- Asignar profesor de forma rotativa (Rotación entre los 4 profesores)
        SET prof_id = SUBSTRING_INDEX(SUBSTRING_INDEX(@profesores, ',', (semester_ua_index % @num_profesores) + 1), ',', -1);

        -- Determinar Patrones de Horario basados en crédito y el índice de rotación
        SET @sesiones = '';
        SET @slots_M = '';
        SET @slots_V = ''; -- Inicializar slots_V
        
        -- 1. Asignación de Patrones (Días y Slots)
        IF ua_credito >= 12.0 OR ua_credito = 3.0 OR ua_credito = 4.5 THEN 
            -- Trabajo Terminal, Estancia, Habilidades Sociales (Baja frecuencia)
            SET @sesiones = @D2_LOW;
            IF semester_ua_index % 2 = 0 THEN
                SET @slots_M = @S2_0; -- T4
                SET @slots_V = 'T3,T3'; -- V3
            ELSE
                SET @slots_M = @S2_1; -- T5
                SET @slots_V = 'T4,T4'; -- V4
            END IF;
            
        ELSEIF ua_credito >= 10.5 THEN
            -- 10.5 créditos (5 sesiones)
            SET @sesiones = @D5;
            IF semester_ua_index % 2 = 0 THEN
                SET @slots_M = @S5_0; -- T1,T2,T3,T4,T5
                SET @slots_V = 'T1,T2,T3,T4,T4'; -- Mapeado a 4 slots vespertinos + 1 duplicado
            ELSE
                 SET @slots_M = @S5_1; -- T5,T1,T2,T3,T4
                 SET @slots_V = 'T4,T1,T2,T3,T4';
            END IF;
            
        ELSEIF ua_credito = 9.0 THEN
            -- 9.0 créditos (4 sesiones)
            SET @sesiones = @D4;
            CASE semester_ua_index % 4 -- Rotación entre 4 patrones de 9.0
                WHEN 0 THEN SET @slots_M = @S4_0; SET @slots_V = 'T1,T2,T3,T4'; -- L1, M2, X3, J4
                WHEN 1 THEN SET @slots_M = @S4_1; SET @slots_V = 'T4,T1,T2,T3'; -- L5, M1, X2, J3
                WHEN 2 THEN SET @slots_M = @S4_2; SET @slots_V = 'T4,T4,T1,T2'; -- L4, M5, X1, J2
                WHEN 3 THEN SET @slots_M = @S4_3; SET @slots_V = 'T3,T4,T4,T1'; -- L3, M4, X5, J1
            END CASE;
            
        ELSEIF ua_credito >= 6.0 AND ua_credito <= 7.5 THEN
            -- 7.5 / 6.0 créditos (3 sesiones)
            SET @sesiones = @D3;
            CASE semester_ua_index % 5 -- Rotación entre 5 patrones de 7.5
                WHEN 0 THEN SET @slots_M = @S3_0; SET @slots_V = 'T1,T2,T3'; -- L1, M2, X3
                WHEN 1 THEN SET @slots_M = @S3_1; SET @slots_V = 'T4,T4,T1'; -- L4, M5, X1
                WHEN 2 THEN SET @slots_M = @S3_2; SET @slots_V = 'T2,T3,T4'; -- L2, M3, X4
                WHEN 3 THEN SET @slots_M = @S3_3; SET @slots_V = 'T4,T1,T2'; -- L5, M1, X2
                WHEN 4 THEN SET @slots_M = @S3_4; SET @slots_V = 'T3,T4,T4'; -- L3, M4, X5
            END CASE;
            
        END IF;

        -- Generar GRUPO MATUTINO (M1)
        SET @grupo_nombre_M = CONCAT(sem_name, 'M1');
        SET @grupo_id_M = CONCAT('G', LPAD(@grupo_id_counter, 4, '0'));
        
        -- Insertar Grupo Matutino
        INSERT INTO grupo (id, nombre, id_ua, id_prof, turno, cupo, reg_final, reg_extra)
        VALUES (@grupo_id_M, @grupo_nombre_M, ua_id_val, prof_id, 'Matutino', 35, NULL, NULL);
        
        SET @grupo_id_counter = @grupo_id_counter + 1;

        -- Generar GRUPO VESPERTINO (V1)
        SET @grupo_nombre_V = CONCAT(sem_name, 'V1');
        SET @grupo_id_V = CONCAT('G', LPAD(@grupo_id_counter, 4, '0'));

        -- Insertar Grupo Vespertino
        INSERT INTO grupo (id, nombre, id_ua, id_prof, turno, cupo, reg_final, reg_extra)
        VALUES (@grupo_id_V, @grupo_nombre_V, ua_id_val, prof_id, 'Vespertino', 35, NULL, NULL);
        
        SET @grupo_id_counter = @grupo_id_counter + 1;

        -- Insertar Distribucion Matutino
        SET @i = 0;
        SET @num_sesiones = LENGTH(@sesiones) - LENGTH(REPLACE(@sesiones, ',', '')) + 1;
        
        WHILE @i < @num_sesiones DO
            SET @dia = SUBSTRING_INDEX(SUBSTRING_INDEX(@sesiones, ',', @i + 1), ',', -1);
            SET @slot_id = SUBSTRING_INDEX(SUBSTRING_INDEX(@slots_M, ',', @i + 1), ',', -1);
            
            -- Obtener el slot de tiempo usando @slot_map_M
            SET @slot_time = SUBSTRING_INDEX(SUBSTRING_INDEX(@slot_map_M, @slot_id, -1), ',', 1);
            SET @hora_full = SUBSTRING_INDEX(@slot_time, '=', -1);
            SET @hora_ini = SUBSTRING_INDEX(@hora_full, '-', 1);
            SET @hora_fin = SUBSTRING_INDEX(@hora_full, '-', -1);
            
            INSERT INTO distribucion (id, id_grupo, hora_ini, hora_fin, dia)
            VALUES (CONCAT('D', LPAD(@dist_id_counter, 4, '0')), @grupo_id_M, @hora_ini, @hora_fin, @dia);
            SET @dist_id_counter = @dist_id_counter + 1;
            SET @i = @i + 1;
        END WHILE;

        -- Insertar Distribucion Vespertino
        SET @i = 0;
        WHILE @i < @num_sesiones DO
            SET @dia = SUBSTRING_INDEX(SUBSTRING_INDEX(@sesiones, ',', @i + 1), ',', -1);
            SET @slot_id = SUBSTRING_INDEX(SUBSTRING_INDEX(@slots_V, ',', @i + 1), ',', -1);
            
            -- Obtener el slot de tiempo usando @slot_map_V
            SET @slot_time = SUBSTRING_INDEX(SUBSTRING_INDEX(@slot_map_V, @slot_id, -1), ',', 1);
            SET @hora_full = SUBSTRING_INDEX(@slot_time, '=', -1);
            SET @hora_ini = SUBSTRING_INDEX(@hora_full, '-', 1);
            SET @hora_fin = SUBSTRING_INDEX(@hora_full, '-', -1);
            
            INSERT INTO distribucion (id, id_grupo, hora_ini, hora_fin, dia)
            VALUES (CONCAT('D', LPAD(@dist_id_counter, 4, '0')), @grupo_id_V, @hora_ini, @hora_fin, @dia);
            SET @dist_id_counter = @dist_id_counter + 1;
            SET @i = @i + 1;
        END WHILE;

        SET semester_ua_index = semester_ua_index + 1; -- Incrementar el índice de rotación dentro del semestre
    END LOOP ua_loop;

    CLOSE ua_cursor;
END $$

DELIMITER ;

CALL generate_groups_and_schedules ();

DROP PROCEDURE generate_groups_and_schedules;

DELETE FROM datos_personales WHERE tipo_usuario = 'alumno';

INSERT INTO
    datos_personales (
        id,
        contrasena,
        tipo_usuario,
        nombre,
        ape_paterno,
        ape_materno,
        fecha_nacimiento,
        RFC,
        tipo_sangre,
        CURP,
        nacionalidad,
        calle,
        num_exterior,
        num_interior,
        codigo_postal,
        colonia,
        delegacion,
        ciudad,
        telefono,
        email,
        grado,
        situacion,
        calificacion,
        carrera
    )
VALUES (
        '2024630101',
        '$2b$10$FVM7xU.TbCeLwfOmDq1QeeF0DCu6FlpuS8f.hqSL55HmZojG1ZbNG',
        'alumno',
        'María',
        'Gómez',
        'Sánchez',
        '2006-03-21',
        'RFC2024630101',
        'A+',
        'CURP2024630101',
        'mexicana',
        'Calle Falsa',
        '123',
        'n/a',
        '01234',
        'Colonia Central',
        'Delegacion X',
        'CDMX',
        '550101',
        '2024630101@ipn.mx',
        NULL,
        'activo',
        8.5,
        'Ingenieria en Inteligencia Artificial'
    ),
    (
        '2024630102',
        '$2b$10$FVM7xU.TbCeLwfOmDq1QeeF0DCu6FlpuS8f.hqSL55HmZojG1ZbNG',
        'alumno',
        'Luis',
        'Hernández',
        'Cruz',
        '2005-11-05',
        'RFC2024630102',
        'O-',
        'CURP2024630102',
        'mexicana',
        'Calle Falsa',
        '123',
        'n/a',
        '01234',
        'Colonia Central',
        'Delegacion X',
        'CDMX',
        '550102',
        '2024630102@ipn.mx',
        NULL,
        'activo',
        8.5,
        'Ingenieria en Inteligencia Artificial'
    ),
    (
        '2024630201',
        '$2b$10$FVM7xU.TbCeLwfOmDq1QeeF0DCu6FlpuS8f.hqSL55HmZojG1ZbNG',
        'alumno',
        'Ana',
        'Díaz',
        'Flores',
        '2006-01-10',
        'RFC2024630201',
        'B+',
        'CURP2024630201',
        'mexicana',
        'Calle Falsa',
        '123',
        'n/a',
        '01234',
        'Colonia Central',
        'Delegacion X',
        'CDMX',
        '550201',
        '2024630201@ipn.mx',
        NULL,
        'activo',
        8.5,
        'Licenciatura en Ciencia de Datos'
    ),
    (
        '2024630202',
        '$2b$10$FVM7xU.TbCeLwfOmDq1QeeF0DCu6FlpuS8f.hqSL55HmZojG1ZbNG',
        'alumno',
        'Pedro',
        'Martínez',
        'López',
        '2005-08-14',
        'RFC2024630202',
        'O+',
        'CURP2024630202',
        'mexicana',
        'Calle Falsa',
        '123',
        'n/a',
        '01234',
        'Colonia Central',
        'Delegacion X',
        'CDMX',
        '550202',
        '2024630202@ipn.mx',
        NULL,
        'activo',
        8.5,
        'Licenciatura en Ciencia de Datos'
    ),
    (
        '2024630301',
        '$2b$10$FVM7xU.TbCeLwfOmDq1QeeF0DCu6FlpuS8f.hqSL55HmZojG1ZbNG',
        'alumno',
        'Sofía',
        'Ramírez',
        'Vargas',
        '2006-05-29',
        'RFC2024630301',
        'A-',
        'CURP2024630301',
        'mexicana',
        'Calle Falsa',
        '123',
        'n/a',
        '01234',
        'Colonia Central',
        'Delegacion X',
        'CDMX',
        '550301',
        '2024630301@ipn.mx',
        NULL,
        'activo',
        8.5,
        'Ingenieria en Sistemas Computacionales'
    ),
    (
        '2024630302',
        '$2b$10$FVM7xU.TbCeLwfOmDq1QeeF0DCu6FlpuS8f.hqSL55HmZojG1ZbNG',
        'alumno',
        'Ricardo',
        'Torres',
        'Pérez',
        '2005-12-01',
        'RFC2024630302',
        'AB+',
        'CURP2024630302',
        'mexicana',
        'Calle Falsa',
        '123',
        'n/a',
        '01234',
        'Colonia Central',
        'Delegacion X',
        'CDMX',
        '550302',
        '2024630302@ipn.mx',
        NULL,
        'activo',
        8.5,
        'Ingenieria en Sistemas Computacionales'
    ),
    (
        '2023630107',
        '$2b$10$FVM7xU.TbCeLwfOmDq1QeeF0DCu6FlpuS8f.hqSL55HmZojG1ZbNG',
        'alumno',
        'Elena',
        'Vega',
        'Ríos',
        '2004-07-19',
        'RFC2023630107',
        'A+',
        'CURP2023630107',
        'mexicana',
        'Calle Falsa',
        '123',
        'n/a',
        '01234',
        'Colonia Central',
        'Delegacion X',
        'CDMX',
        '550107',
        '2023630107@ipn.mx',
        NULL,
        'activo',
        8.5,
        'Ingenieria en Inteligencia Artificial'
    ),
    (
        '2022630208',
        '$2b$10$FVM7xU.TbCeLwfOmDq1QeeF0DCu6FlpuS8f.hqSL55HmZojG1ZbNG',
        'alumno',
        'Javier',
        'Morales',
        'Guzmán',
        '2003-02-02',
        'RFC2022630208',
        'O+',
        'CURP2022630208',
        'mexicana',
        'Calle Falsa',
        '123',
        'n/a',
        '01234',
        'Colonia Central',
        'Delegacion X',
        'CDMX',
        '550208',
        '2022630208@ipn.mx',
        NULL,
        'activo',
        8.5,
        'Licenciatura en Ciencia de Datos'
    ),
    (
        '2021630309',
        '$2b$10$FVM7xU.TbCeLwfOmDq1QeeF0DCu6FlpuS8f.hqSL55HmZojG1ZbNG',
        'alumno',
        'Laura',
        'Castro',
        'Mendoza',
        '2002-09-11',
        'RFC2021630309',
        'B-',
        'CURP2021630309',
        'mexicana',
        'Calle Falsa',
        '123',
        'n/a',
        '01234',
        'Colonia Central',
        'Delegacion X',
        'CDMX',
        '550309',
        '2021630309@ipn.mx',
        NULL,
        'activo',
        8.5,
        'Ingenieria en Sistemas Computacionales'
    ),
    (
        '2021630310',
        '$2b$10$FVM7xU.TbCeLwfOmDq1QeeF0DCu6FlpuS8f.hqSL55HmZojG1ZbNG',
        'alumno',
        'Alejandro',
        'Soto',
        'Núñez',
        '2002-04-25',
        'RFC2021630310',
        'O+',
        'CURP2021630310',
        'mexicana',
        'Calle Falsa',
        '123',
        'n/a',
        '01234',
        'Colonia Central',
        'Delegacion X',
        'CDMX',
        '550310',
        '2021630310@ipn.mx',
        NULL,
        'activo',
        8.5,
        'Ingenieria en Sistemas Computacionales'
    );

-- ==================================================================
-- 5. INSERCIÓN DE ESTUDIANTE Y KARDEX
-- ==================================================================
DELETE FROM estudiante;

INSERT INTO
    estudiante (
        id,
        id_usuario,
        promedio,
        creditos_disponibles,
        estado_academico
    )
VALUES (
        'EST2024630101',
        '2024630101',
        7,
        25.0,
        'Regular'
    ),
    (
        'EST2024630102',
        '2024630102',
        8,
        25.0,
        'Irregular'
    ),
    (
        'EST2024630201',
        '2024630201',
        9.2,
        25.0,
        'Regular'
    ),
    (
        'EST2024630202',
        '2024630202',
        6.5,
        25.0,
        'Regular'
    ),
    (
        'EST2024630301',
        '2024630301',
        8,
        25.0,
        'Irregular'
    ),
    (
        'EST2024630302',
        '2024630302',
        7.7,
        25.0,
        'Irregular'
    ),
    (
        'EST2023630107',
        '2023630107',
        7.7,
        25.0,
        'Regular'
    ),
    (
        'EST2022630208',
        '2022630208',
        8.2,
        25.0,
        'Irregular'
    ),
    (
        'EST2021630309',
        '2021630309',
        8.5,
        25.0,
        'Irregular'
    ),
    (
        'EST2021630310',
        '2021630310',
        9.8,
        25.0,
        'Regular'
    );

DELETE FROM kardex;

INSERT INTO
    kardex (
        id,
        id_alumno,
        promedio,
        situacion_academica,
        semestres_restantes,
        creditos_obtenidos
    )
VALUES (
        'K2024630101',
        '2024630101',
        8.5,
        'Regular',
        1,
        37.5
    ),
    (
        'K2024630102',
        '2024630102',
        8,
        'Irregular',
        5,
        30.0
    ),
    (
        'K2024630201',
        '2024630201',
        9.2,
        'Regular',
        7,
        22.5
    ),
    (
        'K2024630202',
        '2024630202',
        6.5,
        'Regular',
        7,
        45.0
    ),
    (
        'K2024630301',
        '2024630301',
        8,
        'Irregular',
        1,
        52.5
    ),
    (
        'K2024630302',
        '2024630302',
        7.7,
        'Irregular',
        1,
        40.5
    ),
    (
        'K2023630107',
        '2023630107',
        8.6,
        'Irregular',
        5,
        60.0
    ),
    (
        'K2022630208',
        '2022630208',
        8.2,
        'Regular',
        3,
        112.5
    ),
    (
        'K2021630309',
        '2021630309',
        8.5,
        'Irregular',
        1,
        142.5
    ),
    (
        'K2021630310',
        '2021630310',
        9.8,
        'Regular',
        2,
        157.5
    );

-- ==================================================================
-- 6. INSERCIÓN DE UA APROBADA (HISTORIAL ACADÉMICO REALISTA)
-- Se incluye historial para alumnos de 3er semestre en adelante.
-- ==================================================================
DELETE FROM ua_aprobada;

INSERT INTO
    ua_aprobada (
        id,
        id_kardex,
        unidad_aprendizaje,
        calificacion_final,
        semestre,
        periodo,
        fecha,
        metodo_aprobado
    )
VALUES (
        'UAA0001',
        'K2023630107',
        'Fundamentos de Programación',
        8.9,
        1,
        '2024-1',
        '2024-06-20',
        'Ordinario'
    ),
    (
        'UAA0002',
        'K2023630107',
        'Matemáticas Discretas',
        8.8,
        1,
        '2024-1',
        '2024-06-20',
        'Ordinario'
    ),
    (
        'UAA0003',
        'K2023630107',
        'Cálculo',
        7.9,
        1,
        '2024-1',
        '2024-06-20',
        'Ordinario'
    ),
    (
        'UAA0004',
        'K2023630107',
        'Comunicación Oral y Escrita',
        9.4,
        1,
        '2024-1',
        '2024-06-20',
        'Ordinario'
    ),
    (
        'UAA0005',
        'K2023630107',
        'Fundamentos Económicos',
        8.7,
        1,
        '2024-1',
        '2024-06-20',
        'Ordinario'
    ),
    (
        'UAA0006',
        'K2023630107',
        'Algoritmos y Estructuras de Datos',
        9.1,
        2,
        '2024-2',
        '2024-12-20',
        'Ordinario'
    ),
    (
        'UAA0007',
        'K2023630107',
        'Álgebra Lineal',
        8.5,
        2,
        '2024-2',
        '2024-12-20',
        'Ordinario'
    ),
    (
        'UAA0008',
        'K2023630107',
        'Mecánica y Electromagnetismo',
        7.3,
        2,
        '2024-2',
        '2024-12-20',
        'Extraordinario'
    ),
    (
        'UAA0009',
        'K2023630107',
        'Ingeniería, Ética y Sociedad',
        9.8,
        2,
        '2024-2',
        '2024-12-20',
        'Ordinario'
    ),
    (
        'UAA0010',
        'K2023630107',
        'Liderazgo Personal',
        8.0,
        2,
        '2024-2',
        '2024-12-20',
        'Ordinario'
    ),
    (
        'UAA0011',
        'K2022630208',
        'Fundamentos de Programación',
        9.5,
        1,
        '2023-1',
        '2023-06-20',
        'Ordinario'
    ),
    (
        'UAA0012',
        'K2022630208',
        'Matemáticas Discretas',
        9.2,
        1,
        '2023-1',
        '2023-06-20',
        'Ordinario'
    ),
    (
        'UAA0013',
        'K2022630208',
        'Cálculo',
        8.1,
        1,
        '2023-1',
        '2023-06-20',
        'Ordinario'
    ),
    (
        'UAA0014',
        'K2022630208',
        'Comunicación Oral y Escrita',
        9.8,
        1,
        '2023-06-20',
        '2023-06-20',
        'Ordinario'
    ),
    (
        'UAA0015',
        'K2022630208',
        'Introducción a la Ciencia de Datos',
        7.7,
        1,
        '2023-1',
        '2023-06-20',
        'Ordinario'
    ),
    (
        'UAA0016',
        'K2022630208',
        'Algoritmos y Estructuras de Datos',
        8.5,
        2,
        '2023-2',
        '2023-12-20',
        'Ordinario'
    ),
    (
        'UAA0017',
        'K2022630208',
        'Álgebra Lineal',
        7.4,
        2,
        '2023-2',
        '2023-12-20',
        'Extraordinario'
    ),
    (
        'UAA0018',
        'K2022630208',
        'Cálculo Multivariable',
        9.0,
        2,
        '2023-2',
        '2023-12-20',
        'Ordinario'
    ),
    (
        'UAA0019',
        'K2022630208',
        'Ética y Legalidad',
        8.8,
        2,
        '2023-2',
        '2023-12-20',
        'Ordinario'
    ),
    (
        'UAA0020',
        'K2022630208',
        'Fundamentos Económicos',
        8.6,
        2,
        '2023-2',
        '2023-12-20',
        'Ordinario'
    ),
    (
        'UAA0021',
        'K2022630208',
        'Análisis y Diseño de Algoritmos',
        8.1,
        3,
        '2024-1',
        '2024-06-20',
        'Ordinario'
    ),
    (
        'UAA0022',
        'K2022630208',
        'Programación para Ciencia de Datos',
        7.8,
        3,
        '2024-1',
        '2024-06-20',
        'Ordinario'
    ),
    (
        'UAA0023',
        'K2022630208',
        'Probabilidad',
        9.3,
        3,
        '2024-1',
        '2024-06-20',
        'Ordinario'
    ),
    (
        'UAA0024',
        'K2022630208',
        'Bases de Datos',
        8.9,
        3,
        '2024-1',
        '2024-06-20',
        'Ordinario'
    ),
    (
        'UAA0025',
        'K2022630208',
        'Métodos Numéricos',
        8.2,
        3,
        '2024-1',
        '2024-06-20',
        'Ordinario'
    ),
    (
        'UAA0026',
        'K2022630208',
        'Finanzas Empresariales',
        9.7,
        3,
        '2024-1',
        '2024-06-20',
        'Ordinario'
    ),
    (
        'UAA0027',
        'K2022630208',
        'Desarrollo de Aplicaciones Web',
        8.5,
        4,
        '2024-2',
        '2024-12-20',
        'Ordinario'
    ),
    (
        'UAA0028',
        'K2022630208',
        'Cómputo de Alto Desempeño',
        8.1,
        4,
        '2024-2',
        '2024-12-20',
        'Ordinario'
    ),
    (
        'UAA0029',
        'K2022630208',
        'Estadística',
        7.0,
        4,
        '2024-2',
        '2024-12-20',
        'Extraordinario'
    ),
    (
        'UAA0030',
        'K2022630208',
        'Base de Datos Avanzadas',
        8.6,
        4,
        '2024-2',
        '2024-12-20',
        'Ordinario'
    ),
    (
        'UAA0031',
        'K2022630208',
        'Desarrollo de Aplicaciones para Análisis de Datos',
        9.1,
        4,
        '2024-2',
        '2024-12-20',
        'Ordinario'
    ),
    (
        'UAA0032',
        'K2022630208',
        'Liderazgo Personal',
        8.3,
        4,
        '2024-2',
        '2024-12-20',
        'Ordinario'
    ),
    (
        'UAA0033',
        'K2021630309',
        'Fundamentos de Programación',
        8.0,
        1,
        '2022-1',
        '2022-06-20',
        'Ordinario'
    ),
    (
        'UAA0034',
        'K2021630309',
        'Matemáticas Discretas',
        7.5,
        1,
        '2022-1',
        '2022-06-20',
        'Ordinario'
    ),
    (
        'UAA0035',
        'K2021630309',
        'Cálculo',
        9.9,
        1,
        '2022-1',
        '2022-06-20',
        'Ordinario'
    ),
    (
        'UAA0036',
        'K2021630309',
        'Análisis Vectorial',
        8.4,
        1,
        '2022-1',
        '2022-06-20',
        'Ordinario'
    ),
    (
        'UAA0037',
        'K2021630309',
        'Comunicación Oral y Escrita',
        9.2,
        1,
        '2022-1',
        '2022-06-20',
        'Ordinario'
    ),
    (
        'UAA0038',
        'K2021630309',
        'Álgebra Lineal',
        8.7,
        2,
        '2022-2',
        '2022-12-20',
        'Ordinario'
    ),
    (
        'UAA0039',
        'K2021630309',
        'Cálculo Aplicado',
        7.0,
        2,
        '2022-2',
        '2022-12-20',
        'Extraordinario'
    ),
    (
        'UAA0040',
        'K2021630309',
        'Mecánica y Electromagnetismo',
        7.8,
        2,
        '2022-2',
        '2022-12-20',
        'Ordinario'
    ),
    (
        'UAA0041',
        'K2021630309',
        'Ingeniería, Ética y Sociedad',
        9.6,
        2,
        '2022-2',
        '2022-12-20',
        'Ordinario'
    ),
    (
        'UAA0042',
        'K2021630309',
        'Fundamentos Económicos',
        8.3,
        2,
        '2022-2',
        '2022-12-20',
        'Ordinario'
    ),
    (
        'UAA0043',
        'K2021630309',
        'Algoritmos y Estructuras de Datos',
        9.1,
        2,
        '2022-2',
        '2022-12-20',
        'Ordinario'
    ),
    (
        'UAA0044',
        'K2021630309',
        'Ecuaciones Diferenciales',
        8.6,
        3,
        '2023-1',
        '2023-06-20',
        'Ordinario'
    ),
    (
        'UAA0045',
        'K2021630309',
        'Circuitos Eléctricos',
        7.5,
        3,
        '2023-1',
        '2023-06-20',
        'Ordinario'
    ),
    (
        'UAA0046',
        'K2021630309',
        'Fundamentos de Diseño Digital',
        7.2,
        3,
        '2023-1',
        '2023-06-20',
        'Recurse'
    ),
    (
        'UAA0047',
        'K2021630309',
        'Bases de Datos',
        9.0,
        3,
        '2023-1',
        '2023-06-20',
        'Ordinario'
    ),
    (
        'UAA0048',
        'K2021630309',
        'Finanzas Empresariales',
        8.1,
        3,
        '2023-1',
        '2023-06-20',
        'Ordinario'
    ),
    (
        'UAA0049',
        'K2021630309',
        'Paradigmas de Programación',
        7.7,
        3,
        '2023-1',
        '2023-06-20',
        'Ordinario'
    ),
    (
        'UAA0050',
        'K2021630309',
        'Análisis y Diseño de Algoritmos',
        8.5,
        3,
        '2023-1',
        '2023-06-20',
        'Ordinario'
    ),
    (
        'UAA0051',
        'K2021630309',
        'Probabilidad y Estadística',
        8.2,
        4,
        '2023-2',
        '2023-12-20',
        'Ordinario'
    ),
    (
        'UAA0052',
        'K2021630309',
        'Matemáticas Avanzadas para la Ingeniería',
        7.6,
        4,
        '2023-2',
        '2023-12-20',
        'Ordinario'
    ),
    (
        'UAA0053',
        'K2021630309',
        'Electrónica Analógica',
        8.8,
        4,
        '2023-2',
        '2023-12-20',
        'Ordinario'
    ),
    (
        'UAA0054',
        'K2021630309',
        'Diseño de Sistemas Digitales',
        9.3,
        4,
        '2023-2',
        '2023-12-20',
        'Ordinario'
    ),
    (
        'UAA0055',
        'K2021630309',
        'Tecnologías para el Desarrollo de Aplicaciones Web',
        8.0,
        4,
        '2023-2',
        '2023-12-20',
        'Ordinario'
    ),
    (
        'UAA0056',
        'K2021630309',
        'Sistemas Operativos',
        7.5,
        4,
        '2023-2',
        '2023-12-20',
        'Ordinario'
    ),
    (
        'UAA0057',
        'K2021630309',
        'Teoría de la Computación',
        8.7,
        4,
        '2023-2',
        '2023-12-20',
        'Ordinario'
    ),
    (
        'UAA0058',
        'K2021630309',
        'Procesamiento Digital de Señales',
        8.4,
        5,
        '2024-1',
        '2024-06-20',
        'Ordinario'
    ),
    (
        'UAA0059',
        'K2021630309',
        'Instrumentación y Control',
        7.8,
        5,
        '2024-1',
        '2024-06-20',
        'Ordinario'
    ),
    (
        'UAA0060',
        'K2021630309',
        'Arquitectura de Computadoras',
        9.1,
        5,
        '2024-1',
        '2024-06-20',
        'Ordinario'
    ),
    (
        'UAA0061',
        'K2021630309',
        'Análisis y Diseño de Sistemas',
        8.2,
        5,
        '2024-06-20',
        '2024-06-20',
        'Ordinario'
    ),
    (
        'UAA0062',
        'K2021630309',
        'Formulación y Evaluación de Proyectos Informáticos',
        7.0,
        5,
        '2024-1',
        '2024-06-20',
        'Extraordinario'
    ),
    (
        'UAA0063',
        'K2021630309',
        'Compiladores',
        8.9,
        5,
        '2024-1',
        '2024-06-20',
        'Ordinario'
    ),
    (
        'UAA0064',
        'K2021630309',
        'Redes de Computadoras',
        8.5,
        5,
        '2024-1',
        '2024-06-20',
        'Ordinario'
    ),
    (
        'UAA0065',
        'K2021630309',
        'Sistemas en Chip',
        8.7,
        6,
        '2024-2',
        '2024-12-20',
        'Ordinario'
    ),
    (
        'UAA0068',
        'K2021630309',
        'Métodos Cuantitativos para la Toma de Decisiones',
        8.1,
        6,
        '2024-2',
        '2024-12-20',
        'Ordinario'
    ),
    (
        'UAA0069',
        'K2021630309',
        'Ingeniería de Software',
        9.0,
        6,
        '2024-2',
        '2024-12-20',
        'Ordinario'
    ),
    (
        'UAA0070',
        'K2021630309',
        'Inteligencia Artificial',
        7.4,
        6,
        '2024-2',
        '2024-12-20',
        'Extraordinario'
    ),
    (
        'UAA0071',
        'K2021630309',
        'Aplicaciones para Comunicaciones en Red',
        8.6,
        6,
        '2024-2',
        '2024-12-20',
        'Ordinario'
    ),
    -- Alumno 2021630310 (ISC - 8vo semestre) - Historial completo de semestres 1-7
    -- Semestre 1
    (
        'UAA0072',
        'K2021630310',
        'Fundamentos de Programación',
        8.2,
        1,
        '2022-1',
        '2022-06-20',
        'Ordinario'
    ),
    (
        'UAA0073',
        'K2021630310',
        'Matemáticas Discretas',
        7.8,
        1,
        '2022-1',
        '2022-06-20',
        'Ordinario'
    ),
    (
        'UAA0074',
        'K2021630310',
        'Cálculo',
        8.9,
        1,
        '2022-1',
        '2022-06-20',
        'Ordinario'
    ),
    (
        'UAA0075',
        'K2021630310',
        'Análisis Vectorial',
        8.1,
        1,
        '2022-1',
        '2022-06-20',
        'Ordinario'
    ),
    (
        'UAA0076',
        'K2021630310',
        'Comunicación Oral y Escrita',
        9.0,
        1,
        '2022-1',
        '2022-06-20',
        'Ordinario'
    ),
    -- Semestre 2
    (
        'UAA0077',
        'K2021630310',
        'Álgebra Lineal',
        8.6,
        2,
        '2022-2',
        '2022-12-20',
        'Ordinario'
    ),
    (
        'UAA0078',
        'K2021630310',
        'Cálculo Aplicado',
        7.5,
        2,
        '2022-2',
        '2022-12-20',
        'Extraordinario'
    ),
    (
        'UAA0079',
        'K2021630310',
        'Mecánica y Electromagnetismo',
        8.0,
        2,
        '2022-2',
        '2022-12-20',
        'Ordinario'
    ),
    (
        'UAA0080',
        'K2021630310',
        'Ingeniería, Ética y Sociedad',
        9.5,
        2,
        '2022-2',
        '2022-12-20',
        'Ordinario'
    ),
    (
        'UAA0081',
        'K2021630310',
        'Fundamentos Económicos',
        8.4,
        2,
        '2022-2',
        '2022-12-20',
        'Ordinario'
    ),
    (
        'UAA0082',
        'K2021630310',
        'Algoritmos y Estructuras de Datos',
        8.8,
        2,
        '2022-2',
        '2022-12-20',
        'Ordinario'
    ),
    -- Semestre 3
    (
        'UAA0083',
        'K2021630310',
        'Ecuaciones Diferenciales',
        8.3,
        3,
        '2023-1',
        '2023-06-20',
        'Ordinario'
    ),
    (
        'UAA0084',
        'K2021630310',
        'Circuitos Eléctricos',
        7.9,
        3,
        '2023-1',
        '2023-06-20',
        'Ordinario'
    ),
    (
        'UAA0085',
        'K2021630310',
        'Fundamentos de Diseño Digital',
        8.5,
        3,
        '2023-1',
        '2023-06-20',
        'Ordinario'
    ),
    (
        'UAA0086',
        'K2021630310',
        'Bases de Datos',
        9.2,
        3,
        '2023-1',
        '2023-06-20',
        'Ordinario'
    ),
    (
        'UAA0087',
        'K2021630310',
        'Finanzas Empresariales',
        8.7,
        3,
        '2023-1',
        '2023-06-20',
        'Ordinario'
    ),
    (
        'UAA0088',
        'K2021630310',
        'Paradigmas de Programación',
        8.1,
        3,
        '2023-1',
        '2023-06-20',
        'Ordinario'
    ),
    (
        'UAA0089',
        'K2021630310',
        'Análisis y Diseño de Algoritmos',
        9.0,
        3,
        '2023-1',
        '2023-06-20',
        'Ordinario'
    ),
    -- Semestre 4
    (
        'UAA0090',
        'K2021630310',
        'Probabilidad y Estadística',
        8.4,
        4,
        '2023-2',
        '2023-12-20',
        'Ordinario'
    ),
    (
        'UAA0091',
        'K2021630310',
        'Matemáticas Avanzadas para la Ingeniería',
        7.8,
        4,
        '2023-2',
        '2023-12-20',
        'Ordinario'
    ),
    (
        'UAA0092',
        'K2021630310',
        'Electrónica Analógica',
        8.9,
        4,
        '2023-2',
        '2023-12-20',
        'Ordinario'
    ),
    (
        'UAA0093',
        'K2021630310',
        'Diseño de Sistemas Digitales',
        9.1,
        4,
        '2023-2',
        '2023-12-20',
        'Ordinario'
    ),
    (
        'UAA0094',
        'K2021630310',
        'Tecnologías para el Desarrollo de Aplicaciones Web',
        8.6,
        4,
        '2023-2',
        '2023-12-20',
        'Ordinario'
    ),
    (
        'UAA0095',
        'K2021630310',
        'Sistemas Operativos',
        8.2,
        4,
        '2023-2',
        '2023-12-20',
        'Ordinario'
    ),
    (
        'UAA0096',
        'K2021630310',
        'Teoría de la Computación',
        9.0,
        4,
        '2023-2',
        '2023-12-20',
        'Ordinario'
    ),
    -- Semestre 5
    (
        'UAA0097',
        'K2021630310',
        'Procesamiento Digital de Señales',
        8.7,
        5,
        '2024-1',
        '2024-06-20',
        'Ordinario'
    ),
    (
        'UAA0098',
        'K2021630310',
        'Instrumentación y Control',
        8.0,
        5,
        '2024-1',
        '2024-06-20',
        'Ordinario'
    ),
    (
        'UAA0099',
        'K2021630310',
        'Arquitectura de Computadoras',
        9.3,
        5,
        '2024-1',
        '2024-06-20',
        'Ordinario'
    ),
    (
        'UAA0100',
        'K2021630310',
        'Análisis y Diseño de Sistemas',
        8.5,
        5,
        '2024-1',
        '2024-06-20',
        'Ordinario'
    ),
    (
        'UAA0101',
        'K2021630310',
        'Formulación y Evaluación de Proyectos Informáticos',
        7.2,
        5,
        '2024-1',
        '2024-06-20',
        'Extraordinario'
    ),
    (
        'UAA0102',
        'K2021630310',
        'Compiladores',
        8.8,
        5,
        '2024-1',
        '2024-06-20',
        'Ordinario'
    ),
    (
        'UAA0103',
        'K2021630310',
        'Redes de Computadoras',
        8.4,
        5,
        '2024-1',
        '2024-06-20',
        'Ordinario'
    ),
    -- Semestre 6
    (
        'UAA0104',
        'K2021630310',
        'Sistemas en Chip',
        8.6,
        6,
        '2024-2',
        '2024-12-20',
        'Ordinario'
    ),
    (
        'UAA0105',
        'K2021630310',
        'Métodos Cuantitativos para la Toma de Decisiones',
        8.3,
        6,
        '2024-2',
        '2024-12-20',
        'Ordinario'
    ),
    (
        'UAA0106',
        'K2021630310',
        'Ingeniería de Software',
        9.1,
        6,
        '2024-2',
        '2024-12-20',
        'Ordinario'
    ),
    (
        'UAA0107',
        'K2021630310',
        'Inteligencia Artificial',
        8.0,
        6,
        '2024-2',
        '2024-12-20',
        'Ordinario'
    ),
    (
        'UAA0108',
        'K2021630310',
        'Aplicaciones para Comunicaciones en Red',
        8.9,
        6,
        '2024-2',
        '2024-12-20',
        'Ordinario'
    ),
    (
        'UAA0109',
        'K2021630310',
        'Seguridad Informática',
        9.0,
        6,
        '2024-2',
        '2024-12-20',
        'Ordinario'
    ),
    -- Semestre 7
    (
        'UAA0111',
        'K2021630310',
        'Desarrollo de Aplicaciones Móviles Nativas',
        8.4,
        7,
        '2025-1',
        '2025-06-20',
        'Ordinario'
    ),
    (
        'UAA0114',
        'K2021630310',
        'Sistemas Distribuidos',
        8.2,
        7,
        '2025-1',
        '2025-06-20',
        'Ordinario'
    ),
    (
        'UAA0115',
        'K2021630310',
        'Administración de Servicios en Red',
        8.7,
        7,
        '2025-1',
        '2025-06-20',
        'Ordinario'
    );

-- ==================================================================
-- 7. INSCRIPCIONES DEL SEMESTRE ACTUAL (HORARIOS Y MATRÍCULAS)
-- Los alumnos están inscritos en un grupo Matutino sin traslapes.
-- ==================================================================
DELETE FROM horario;

INSERT INTO
    horario (id, id_alumno)
VALUES ('H2024630101', '2024630101'),
    ('H2024630102', '2024630102'),
    ('H2024630201', '2024630201'),
    ('H2024630202', '2024630202'),
    ('H2024630301', '2024630301'),
    ('H2024630302', '2024630302'),
    ('H2023630107', '2023630107'),
    ('H2022630208', '2022630208'),
    ('H2021630309', '2021630309'),
    ('H2021630310', '2021630310');

DELETE FROM inscripcion;

INSERT INTO
    inscripcion (
        id,
        id_alumno,
        fecha_hora_in,
        fecha_hora_cad
    )
VALUES (
        'I2024630101',
        '2024630101',
        NOW(),
        DATE_ADD(NOW(), INTERVAL 30 DAY)
    ),
    (
        'I2024630102',
        '2024630102',
        NOW(),
        DATE_ADD(NOW(), INTERVAL 30 DAY)
    ),
    (
        'I2024630201',
        '2024630201',
        NOW(),
        DATE_ADD(NOW(), INTERVAL 30 DAY)
    ),
    (
        'I2024630202',
        '2024630202',
        NOW(),
        DATE_ADD(NOW(), INTERVAL 30 DAY)
    ),
    (
        'I2024630301',
        '2024630301',
        NOW(),
        DATE_ADD(NOW(), INTERVAL 30 DAY)
    ),
    (
        'I2024630302',
        '2024630302',
        NOW(),
        DATE_ADD(NOW(), INTERVAL 30 DAY)
    ),
    (
        'I2023630107',
        '2023630107',
        NOW(),
        DATE_ADD(NOW(), INTERVAL 30 DAY)
    ),
    (
        'I2022630208',
        '2022630208',
        NOW(),
        DATE_ADD(NOW(), INTERVAL 30 DAY)
    ),
    (
        'I2021630309',
        '2021630309',
        NOW(),
        DATE_ADD(NOW(), INTERVAL 30 DAY)
    ),
    (
        'I2021630310',
        '2021630310',
        NOW(),
        DATE_ADD(NOW(), INTERVAL 30 DAY)
    );

DELETE FROM mat_inscritos;

INSERT INTO
    mat_inscritos (
        id,
        id_horario,
        id_grupo,
        calificacion_final
    )
VALUES
    -- 1BM1 (IIA) Alumnos: 2024630101, 2024630102
    (
        'M0001',
        'H2024630101',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1BM1'
                AND id_ua = 'UA0051'
        ),
        NULL
    ),
    (
        'M0002',
        'H2024630101',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1BM1'
                AND id_ua = 'UA0052'
        ),
        NULL
    ),
    (
        'M0003',
        'H2024630101',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1BM1'
                AND id_ua = 'UA0053'
        ),
        NULL
    ),
    (
        'M0004',
        'H2024630101',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1BM1'
                AND id_ua = 'UA0054'
        ),
        NULL
    ),
    (
        'M0005',
        'H2024630101',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1BM1'
                AND id_ua = 'UA0055'
        ),
        NULL
    ),
    (
        'M0006',
        'H2024630102',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1BM1'
                AND id_ua = 'UA0051'
        ),
        NULL
    ),
    (
        'M0007',
        'H2024630102',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1BM1'
                AND id_ua = 'UA0052'
        ),
        NULL
    ),
    (
        'M0008',
        'H2024630102',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1BM1'
                AND id_ua = 'UA0053'
        ),
        NULL
    ),
    (
        'M0009',
        'H2024630102',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1BM1'
                AND id_ua = 'UA0054'
        ),
        NULL
    ),
    (
        'M0010',
        'H2024630102',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1BM1'
                AND id_ua = 'UA0055'
        ),
        NULL
    ),
    -- 1AM1 (LCD) Alumnos: 2024630201, 2024630202
    (
        'M0011',
        'H2024630201',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1AM1'
                AND id_ua = 'UA0096'
        ),
        NULL
    ),
    (
        'M0012',
        'H2024630201',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1AM1'
                AND id_ua = 'UA0097'
        ),
        NULL
    ),
    (
        'M0013',
        'H2024630201',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1AM1'
                AND id_ua = 'UA0098'
        ),
        NULL
    ),
    (
        'M0014',
        'H2024630201',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1AM1'
                AND id_ua = 'UA0099'
        ),
        NULL
    ),
    (
        'M0015',
        'H2024630201',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1AM1'
                AND id_ua = 'UA0100'
        ),
        NULL
    ),
    (
        'M0016',
        'H2024630202',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1AM1'
                AND id_ua = 'UA0096'
        ),
        NULL
    ),
    (
        'M0017',
        'H2024630202',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1AM1'
                AND id_ua = 'UA0097'
        ),
        NULL
    ),
    (
        'M0018',
        'H2024630202',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1AM1'
                AND id_ua = 'UA0098'
        ),
        NULL
    ),
    (
        'M0019',
        'H2024630202',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1AM1'
                AND id_ua = 'UA0099'
        ),
        NULL
    ),
    (
        'M0020',
        'H2024630202',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1AM1'
                AND id_ua = 'UA0100'
        ),
        NULL
    ),
    -- 1CM1 (ISC) Alumnos: 2024630301, 2024630302
    (
        'M0021',
        'H2024630301',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1CM1'
                AND id_ua = 'UA0001'
        ),
        NULL
    ),
    (
        'M0022',
        'H2024630301',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1CM1'
                AND id_ua = 'UA0002'
        ),
        NULL
    ),
    (
        'M0023',
        'H2024630301',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1CM1'
                AND id_ua = 'UA0003'
        ),
        NULL
    ),
    (
        'M0024',
        'H2024630301',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1CM1'
                AND id_ua = 'UA0004'
        ),
        NULL
    ),
    (
        'M0025',
        'H2024630301',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1CM1'
                AND id_ua = 'UA0005'
        ),
        NULL
    ),
    (
        'M0026',
        'H2024630302',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1CM1'
                AND id_ua = 'UA0001'
        ),
        NULL
    ),
    (
        'M0027',
        'H2024630302',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1CM1'
                AND id_ua = 'UA0002'
        ),
        NULL
    ),
    (
        'M0028',
        'H2024630302',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1CM1'
                AND id_ua = 'UA0003'
        ),
        NULL
    ),
    (
        'M0029',
        'H2024630302',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1CM1'
                AND id_ua = 'UA0004'
        ),
        NULL
    ),
    (
        'M0030',
        'H2024630302',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '1CM1'
                AND id_ua = 'UA0005'
        ),
        NULL
    ),
    -- 3BM1 (IIA) Alumno: 2023630107
    (
        'M0031',
        'H2023630107',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '3BM1'
                AND id_ua = 'UA0061'
        ),
        NULL
    ),
    (
        'M0032',
        'H2023630107',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '3BM1'
                AND id_ua = 'UA0062'
        ),
        NULL
    ),
    (
        'M0033',
        'H2023630107',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '3BM1'
                AND id_ua = 'UA0063'
        ),
        NULL
    ),
    (
        'M0034',
        'H2023630107',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '3BM1'
                AND id_ua = 'UA0064'
        ),
        NULL
    ),
    (
        'M0035',
        'H2023630107',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '3BM1'
                AND id_ua = 'UA0065'
        ),
        NULL
    ),
    (
        'M0036',
        'H2023630107',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '3BM1'
                AND id_ua = 'UA0066'
        ),
        NULL
    ),
    -- 5AM1 (LCD) Alumno: 2022630208
    (
        'M0037',
        'H2022630208',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '5AM1'
                AND id_ua = 'UA0118'
        ),
        NULL
    ),
    (
        'M0038',
        'H2022630208',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '5AM1'
                AND id_ua = 'UA0119'
        ),
        NULL
    ),
    (
        'M0039',
        'H2022630208',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '5AM1'
                AND id_ua = 'UA0120'
        ),
        NULL
    ),
    (
        'M0040',
        'H2022630208',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '5AM1'
                AND id_ua = 'UA0121'
        ),
        NULL
    ),
    (
        'M0041',
        'H2022630208',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '5AM1'
                AND id_ua = 'UA0122'
        ),
        NULL
    ),
    (
        'M0042',
        'H2022630208',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '5AM1'
                AND id_ua = 'UA0123'
        ),
        NULL
    ),
    -- 7CM1 (ISC) Alumno: 2021630309
    (
        'M0043',
        'H2021630309',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '7CM1'
                AND id_ua = 'UA0040'
        ),
        NULL
    ),
    (
        'M0044',
        'H2021630309',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '7CM1'
                AND id_ua = 'UA0043'
        ),
        NULL
    ),
    (
        'M0045',
        'H2021630309',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '7CM1'
                AND id_ua = 'UA0044'
        ),
        NULL
    ),
    -- 8CM1 (ISC) Alumno: 2021630310
    (
        'M0046',
        'H2021630310',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '8CM1'
                AND id_ua = 'UA0045'
        ),
        NULL
    ),
    (
        'M0047',
        'H2021630310',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '8CM1'
                AND id_ua = 'UA0046'
        ),
        NULL
    ),
    (
        'M0048',
        'H2021630310',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '8CM1'
                AND id_ua = 'UA0047'
        ),
        NULL
    ),
    (
        'M0049',
        'H2021630310',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '8CM1'
                AND id_ua = 'UA0048'
        ),
        NULL
    ),
    (
        'M0050',
        'H2021630310',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '8CM1'
                AND id_ua = 'UA0049'
        ),
        NULL
    ),
    (
        'M0051',
        'H2021630310',
        (
            SELECT id
            FROM grupo
            WHERE
                nombre = '8CM1'
                AND id_ua = 'UA0050'
        ),
        NULL
    );

-- ==================================================================
-- GRUPOS PARA MATERIAS OPTATIVAS
-- Se agregan grupos para materias OPTATIVAS de cada carrera
-- ==================================================================

-- OPTATIVAS ISC (Ingenieria en Sistemas Computacionales) - Semestre 6
INSERT INTO
    grupo (
        id,
        nombre,
        id_ua,
        id_prof,
        turno,
        cupo,
        reg_final,
        reg_extra
    )
VALUES (
        'G_OPT_001',
        '6CM1',
        'UA0140',
        'HIJKLMNO',
        'Matutino',
        30,
        0,
        0
    ),
    (
        'G_OPT_002',
        '6CM2',
        'UA0141',
        'P0001RSV',
        'Matutino',
        30,
        0,
        0
    ),
    (
        'G_OPT_003',
        '6CM3',
        'UA0142',
        'P0002KND',
        'Vespertino',
        30,
        0,
        0
    ),
    (
        'G_OPT_004',
        '6CM4',
        'UA0143',
        'P0003HGR',
        'Vespertino',
        30,
        0,
        0
    ),
    (
        'G_OPT_005',
        '6CM5',
        'UA0144',
        'HIJKLMNO',
        'Matutino',
        30,
        0,
        0
    ),
    (
        'G_OPT_006',
        '6CM6',
        'UA0145',
        'P0001RSV',
        'Vespertino',
        30,
        0,
        0
    ),
    (
        'G_OPT_007',
        '6CM7',
        'UA0146',
        'P0002KND',
        'Matutino',
        30,
        0,
        0
    ),
    (
        'G_OPT_008',
        '6CM8',
        'UA0147',
        'P0003HGR',
        'Vespertino',
        30,
        0,
        0
    ),
    (
        'G_OPT_009',
        '6CM9',
        'UA0148',
        'HIJKLMNO',
        'Matutino',
        30,
        0,
        0
    ),
    (
        'G_OPT_010',
        '6CM10',
        'UA0149',
        'P0001RSV',
        'Vespertino',
        30,
        0,
        0
    ),
    (
        'G_OPT_011',
        '6CM11',
        'UA0150',
        'P0002KND',
        'Matutino',
        30,
        0,
        0
    ),
    (
        'G_OPT_012',
        '6CM12',
        'UA0151',
        'P0003HGR',
        'Vespertino',
        30,
        0,
        0
    ),
    (
        'G_OPT_013',
        '6CM13',
        'UA0152',
        'HIJKLMNO',
        'Matutino',
        30,
        0,
        0
    ),
    (
        'G_OPT_014',
        '6CM14',
        'UA0153',
        'P0001RSV',
        'Vespertino',
        30,
        0,
        0
    );

-- OPTATIVAS ISC - Semestre 7
INSERT INTO
    grupo (
        id,
        nombre,
        id_ua,
        id_prof,
        turno,
        cupo,
        reg_final,
        reg_extra
    )
VALUES (
        'G_OPT_015',
        '7CM1',
        'UA0154',
        'P0002KND',
        'Matutino',
        30,
        0,
        0
    ),
    (
        'G_OPT_016',
        '7CM2',
        'UA0155',
        'P0003HGR',
        'Vespertino',
        30,
        0,
        0
    ),
    (
        'G_OPT_017',
        '7CM3',
        'UA0156',
        'HIJKLMNO',
        'Matutino',
        30,
        0,
        0
    ),
    (
        'G_OPT_018',
        '7CM4',
        'UA0157',
        'P0001RSV',
        'Vespertino',
        30,
        0,
        0
    ),
    (
        'G_OPT_019',
        '7CM5',
        'UA0158',
        'P0002KND',
        'Matutino',
        30,
        0,
        0
    ),
    (
        'G_OPT_020',
        '7CM6',
        'UA0159',
        'P0003HGR',
        'Vespertino',
        30,
        0,
        0
    ),
    (
        'G_OPT_021',
        '7CM7',
        'UA0160',
        'HIJKLMNO',
        'Matutino',
        30,
        0,
        0
    ),
    (
        'G_OPT_022',
        '7CM8',
        'UA0161',
        'P0001RSV',
        'Vespertino',
        30,
        0,
        0
    ),
    (
        'G_OPT_023',
        '7CM9',
        'UA0162',
        'P0002KND',
        'Matutino',
        30,
        0,
        0
    ),
    (
        'G_OPT_024',
        '7CM10',
        'UA0163',
        'P0003HGR',
        'Vespertino',
        30,
        0,
        0
    );

-- OPTATIVAS IIA (Ingenieria en Inteligencia Artificial) - Semestre 6
INSERT INTO
    grupo (
        id,
        nombre,
        id_ua,
        id_prof,
        turno,
        cupo,
        reg_final,
        reg_extra
    )
VALUES (
        'G_OPT_025',
        '6BM1',
        'UA0164',
        'HIJKLMNO',
        'Matutino',
        30,
        0,
        0
    ),
    (
        'G_OPT_026',
        '6BM2',
        'UA0165',
        'P0001RSV',
        'Vespertino',
        30,
        0,
        0
    ),
    (
        'G_OPT_027',
        '6BM3',
        'UA0166',
        'P0002KND',
        'Matutino',
        30,
        0,
        0
    ),
    (
        'G_OPT_028',
        '6BM4',
        'UA0167',
        'P0003HGR',
        'Vespertino',
        30,
        0,
        0
    ),
    (
        'G_OPT_029',
        '6BM5',
        'UA0168',
        'HIJKLMNO',
        'Matutino',
        30,
        0,
        0
    ),
    (
        'G_OPT_030',
        '6BM6',
        'UA0169',
        'P0001RSV',
        'Vespertino',
        30,
        0,
        0
    ),
    (
        'G_OPT_031',
        '6BM7',
        'UA0170',
        'P0002KND',
        'Matutino',
        30,
        0,
        0
    );

-- OPTATIVAS IIA - Semestre 7
INSERT INTO
    grupo (
        id,
        nombre,
        id_ua,
        id_prof,
        turno,
        cupo,
        reg_final,
        reg_extra
    )
VALUES (
        'G_OPT_032',
        '7BM1',
        'UA0171',
        'P0003HGR',
        'Vespertino',
        30,
        0,
        0
    ),
    (
        'G_OPT_033',
        '7BM2',
        'UA0172',
        'HIJKLMNO',
        'Matutino',
        30,
        0,
        0
    ),
    (
        'G_OPT_034',
        '7BM3',
        'UA0173',
        'P0001RSV',
        'Vespertino',
        30,
        0,
        0
    ),
    (
        'G_OPT_035',
        '7BM4',
        'UA0174',
        'P0002KND',
        'Matutino',
        30,
        0,
        0
    ),
    (
        'G_OPT_036',
        '7BM5',
        'UA0175',
        'P0003HGR',
        'Vespertino',
        30,
        0,
        0
    ),
    (
        'G_OPT_037',
        '7BM6',
        'UA0176',
        'HIJKLMNO',
        'Matutino',
        30,
        0,
        0
    ),
    (
        'G_OPT_038',
        '7BM7',
        'UA0177',
        'P0001RSV',
        'Vespertino',
        30,
        0,
        0
    );

-- OPTATIVAS LCD (Licenciatura en Ciencia de Datos) - Semestre 6
INSERT INTO
    grupo (
        id,
        nombre,
        id_ua,
        id_prof,
        turno,
        cupo,
        reg_final,
        reg_extra
    )
VALUES (
        'G_OPT_039',
        '6AM1',
        'UA0178',
        'P0002KND',
        'Matutino',
        30,
        0,
        0
    ),
    (
        'G_OPT_040',
        '6AM2',
        'UA0179',
        'P0003HGR',
        'Vespertino',
        30,
        0,
        0
    ),
    (
        'G_OPT_041',
        '6AM3',
        'UA0180',
        'HIJKLMNO',
        'Matutino',
        30,
        0,
        0
    ),
    (
        'G_OPT_042',
        '6AM4',
        'UA0181',
        'P0001RSV',
        'Vespertino',
        30,
        0,
        0
    ),
    (
        'G_OPT_043',
        '6AM5',
        'UA0182',
        'P0002KND',
        'Matutino',
        30,
        0,
        0
    ),
    (
        'G_OPT_044',
        '6AM6',
        'UA0183',
        'P0003HGR',
        'Vespertino',
        30,
        0,
        0
    );

-- OPTATIVAS LCD - Semestre 7
INSERT INTO
    grupo (
        id,
        nombre,
        id_ua,
        id_prof,
        turno,
        cupo,
        reg_final,
        reg_extra
    )
VALUES (
        'G_OPT_045',
        '7AM1',
        'UA0184',
        'HIJKLMNO',
        'Matutino',
        30,
        0,
        0
    ),
    (
        'G_OPT_046',
        '7AM2',
        'UA0185',
        'P0001RSV',
        'Vespertino',
        30,
        0,
        0
    ),
    (
        'G_OPT_047',
        '7AM3',
        'UA0186',
        'P0002KND',
        'Matutino',
        30,
        0,
        0
    ),
    (
        'G_OPT_048',
        '7AM4',
        'UA0187',
        'P0003HGR',
        'Vespertino',
        30,
        0,
        0
    ),
    (
        'G_OPT_049',
        '7AM5',
        'UA0188',
        'HIJKLMNO',
        'Matutino',
        30,
        0,
        0
    ),
    (
        'G_OPT_050',
        '7AM6',
        'UA0189',
        'P0001RSV',
        'Vespertino',
        30,
        0,
        0
    ),
    (
        'G_OPT_051',
        '7AM7',
        'UA0190',
        'P0002KND',
        'Matutino',
        30,
        0,
        0
    );

-- DISTRIBUCIÓN DE HORARIOS PARA GRUPOS OPTATIVAS
-- Horario estándar: Lunes, Miércoles, Viernes 10:30-12:00 (Matutino) o 6:30-8:00 (Vespertino)
INSERT INTO
    distribucion (
        id,
        id_grupo,
        hora_ini,
        hora_fin,
        dia
    )
VALUES (
        'D_OPT_001',
        'G_OPT_001',
        '10:30',
        '12:00',
        'Lunes'
    ),
    (
        'D_OPT_002',
        'G_OPT_001',
        '10:30',
        '12:00',
        'Miércoles'
    ),
    (
        'D_OPT_003',
        'G_OPT_001',
        '10:30',
        '12:00',
        'Viernes'
    ),
    (
        'D_OPT_004',
        'G_OPT_002',
        '08:30',
        '10:00',
        'Lunes'
    ),
    (
        'D_OPT_005',
        'G_OPT_002',
        '08:30',
        '10:00',
        'Miércoles'
    ),
    (
        'D_OPT_006',
        'G_OPT_002',
        '08:30',
        '10:00',
        'Viernes'
    ),
    (
        'D_OPT_007',
        'G_OPT_005',
        '12:00',
        '13:30',
        'Lunes'
    ),
    (
        'D_OPT_008',
        'G_OPT_005',
        '12:00',
        '13:30',
        'Miércoles'
    ),
    (
        'D_OPT_009',
        'G_OPT_005',
        '12:00',
        '13:30',
        'Viernes'
    ),
    (
        'D_OPT_010',
        'G_OPT_007',
        '07:00',
        '08:30',
        'Martes'
    ),
    (
        'D_OPT_011',
        'G_OPT_007',
        '07:00',
        '08:30',
        'Jueves'
    ),
    (
        'D_OPT_012',
        'G_OPT_007',
        '07:00',
        '08:30',
        'Viernes'
    ),
    (
        'D_OPT_013',
        'G_OPT_009',
        '13:30',
        '15:00',
        'Lunes'
    ),
    (
        'D_OPT_014',
        'G_OPT_009',
        '13:30',
        '15:00',
        'Miércoles'
    ),
    (
        'D_OPT_015',
        'G_OPT_009',
        '13:30',
        '15:00',
        'Viernes'
    ),
    (
        'D_OPT_016',
        'G_OPT_011',
        '10:30',
        '12:00',
        'Martes'
    ),
    (
        'D_OPT_017',
        'G_OPT_011',
        '10:30',
        '12:00',
        'Jueves'
    ),
    (
        'D_OPT_018',
        'G_OPT_011',
        '10:30',
        '12:00',
        'Viernes'
    ),
    (
        'D_OPT_019',
        'G_OPT_013',
        '08:30',
        '10:00',
        'Martes'
    ),
    (
        'D_OPT_020',
        'G_OPT_013',
        '08:30',
        '10:00',
        'Jueves'
    ),
    (
        'D_OPT_021',
        'G_OPT_013',
        '08:30',
        '10:00',
        'Viernes'
    );

-- ISC Semestre 6 - Vespertino
INSERT INTO
    distribucion (
        id,
        id_grupo,
        hora_ini,
        hora_fin,
        dia
    )
VALUES (
        'D_OPT_022',
        'G_OPT_003',
        '18:30',
        '20:00',
        'Lunes'
    ),
    (
        'D_OPT_023',
        'G_OPT_003',
        '18:30',
        '20:00',
        'Miércoles'
    ),
    (
        'D_OPT_024',
        'G_OPT_003',
        '18:30',
        '20:00',
        'Viernes'
    ),
    (
        'D_OPT_025',
        'G_OPT_004',
        '17:00',
        '18:30',
        'Lunes'
    ),
    (
        'D_OPT_026',
        'G_OPT_004',
        '17:00',
        '18:30',
        'Miércoles'
    ),
    (
        'D_OPT_027',
        'G_OPT_004',
        '17:00',
        '18:30',
        'Viernes'
    ),
    (
        'D_OPT_028',
        'G_OPT_006',
        '15:00',
        '16:30',
        'Lunes'
    ),
    (
        'D_OPT_029',
        'G_OPT_006',
        '15:00',
        '16:30',
        'Miércoles'
    ),
    (
        'D_OPT_030',
        'G_OPT_006',
        '15:00',
        '16:30',
        'Viernes'
    ),
    (
        'D_OPT_031',
        'G_OPT_008',
        '20:00',
        '21:30',
        'Martes'
    ),
    (
        'D_OPT_032',
        'G_OPT_008',
        '20:00',
        '21:30',
        'Jueves'
    ),
    (
        'D_OPT_033',
        'G_OPT_008',
        '20:00',
        '21:30',
        'Viernes'
    ),
    (
        'D_OPT_034',
        'G_OPT_010',
        '18:30',
        '20:00',
        'Martes'
    ),
    (
        'D_OPT_035',
        'G_OPT_010',
        '18:30',
        '20:00',
        'Jueves'
    ),
    (
        'D_OPT_036',
        'G_OPT_010',
        '18:30',
        '20:00',
        'Viernes'
    ),
    (
        'D_OPT_037',
        'G_OPT_012',
        '17:00',
        '18:30',
        'Martes'
    ),
    (
        'D_OPT_038',
        'G_OPT_012',
        '17:00',
        '18:30',
        'Jueves'
    ),
    (
        'D_OPT_039',
        'G_OPT_012',
        '17:00',
        '18:30',
        'Viernes'
    ),
    (
        'D_OPT_040',
        'G_OPT_014',
        '15:00',
        '16:30',
        'Martes'
    ),
    (
        'D_OPT_041',
        'G_OPT_014',
        '15:00',
        '16:30',
        'Jueves'
    ),
    (
        'D_OPT_042',
        'G_OPT_014',
        '15:00',
        '16:30',
        'Viernes'
    );

-- ISC Semestre 7 - Matutino
INSERT INTO
    distribucion (
        id,
        id_grupo,
        hora_ini,
        hora_fin,
        dia
    )
VALUES (
        'D_OPT_043',
        'G_OPT_015',
        '10:30',
        '12:00',
        'Lunes'
    ),
    (
        'D_OPT_044',
        'G_OPT_015',
        '10:30',
        '12:00',
        'Miércoles'
    ),
    (
        'D_OPT_045',
        'G_OPT_015',
        '10:30',
        '12:00',
        'Viernes'
    ),
    (
        'D_OPT_046',
        'G_OPT_017',
        '08:30',
        '10:00',
        'Lunes'
    ),
    (
        'D_OPT_047',
        'G_OPT_017',
        '08:30',
        '10:00',
        'Miércoles'
    ),
    (
        'D_OPT_048',
        'G_OPT_017',
        '08:30',
        '10:00',
        'Viernes'
    ),
    (
        'D_OPT_049',
        'G_OPT_019',
        '12:00',
        '13:30',
        'Lunes'
    ),
    (
        'D_OPT_050',
        'G_OPT_019',
        '12:00',
        '13:30',
        'Miércoles'
    ),
    (
        'D_OPT_051',
        'G_OPT_019',
        '12:00',
        '13:30',
        'Viernes'
    ),
    (
        'D_OPT_052',
        'G_OPT_021',
        '07:00',
        '08:30',
        'Martes'
    ),
    (
        'D_OPT_053',
        'G_OPT_021',
        '07:00',
        '08:30',
        'Jueves'
    ),
    (
        'D_OPT_054',
        'G_OPT_021',
        '07:00',
        '08:30',
        'Viernes'
    ),
    (
        'D_OPT_055',
        'G_OPT_023',
        '13:30',
        '15:00',
        'Lunes'
    ),
    (
        'D_OPT_056',
        'G_OPT_023',
        '13:30',
        '15:00',
        'Miércoles'
    ),
    (
        'D_OPT_057',
        'G_OPT_023',
        '13:30',
        '15:00',
        'Viernes'
    );

-- ISC Semestre 7 - Vespertino
INSERT INTO
    distribucion (
        id,
        id_grupo,
        hora_ini,
        hora_fin,
        dia
    )
VALUES (
        'D_OPT_058',
        'G_OPT_016',
        '18:30',
        '20:00',
        'Lunes'
    ),
    (
        'D_OPT_059',
        'G_OPT_016',
        '18:30',
        '20:00',
        'Miércoles'
    ),
    (
        'D_OPT_060',
        'G_OPT_016',
        '18:30',
        '20:00',
        'Viernes'
    ),
    (
        'D_OPT_061',
        'G_OPT_018',
        '17:00',
        '18:30',
        'Lunes'
    ),
    (
        'D_OPT_062',
        'G_OPT_018',
        '17:00',
        '18:30',
        'Miércoles'
    ),
    (
        'D_OPT_063',
        'G_OPT_018',
        '17:00',
        '18:30',
        'Viernes'
    ),
    (
        'D_OPT_064',
        'G_OPT_020',
        '15:00',
        '16:30',
        'Lunes'
    ),
    (
        'D_OPT_065',
        'G_OPT_020',
        '15:00',
        '16:30',
        'Miércoles'
    ),
    (
        'D_OPT_066',
        'G_OPT_020',
        '15:00',
        '16:30',
        'Viernes'
    ),
    (
        'D_OPT_067',
        'G_OPT_022',
        '20:00',
        '21:30',
        'Martes'
    ),
    (
        'D_OPT_068',
        'G_OPT_022',
        '20:00',
        '21:30',
        'Jueves'
    ),
    (
        'D_OPT_069',
        'G_OPT_022',
        '20:00',
        '21:30',
        'Viernes'
    ),
    (
        'D_OPT_070',
        'G_OPT_024',
        '18:30',
        '20:00',
        'Martes'
    ),
    (
        'D_OPT_071',
        'G_OPT_024',
        '18:30',
        '20:00',
        'Jueves'
    ),
    (
        'D_OPT_072',
        'G_OPT_024',
        '18:30',
        '20:00',
        'Viernes'
    );

-- IIA Semestre 6 - Matutino
INSERT INTO
    distribucion (
        id,
        id_grupo,
        hora_ini,
        hora_fin,
        dia
    )
VALUES (
        'D_OPT_073',
        'G_OPT_025',
        '10:30',
        '12:00',
        'Lunes'
    ),
    (
        'D_OPT_074',
        'G_OPT_025',
        '10:30',
        '12:00',
        'Miércoles'
    ),
    (
        'D_OPT_075',
        'G_OPT_025',
        '10:30',
        '12:00',
        'Viernes'
    ),
    (
        'D_OPT_076',
        'G_OPT_027',
        '08:30',
        '10:00',
        'Lunes'
    ),
    (
        'D_OPT_077',
        'G_OPT_027',
        '08:30',
        '10:00',
        'Miércoles'
    ),
    (
        'D_OPT_078',
        'G_OPT_027',
        '08:30',
        '10:00',
        'Viernes'
    ),
    (
        'D_OPT_079',
        'G_OPT_029',
        '12:00',
        '13:30',
        'Lunes'
    ),
    (
        'D_OPT_080',
        'G_OPT_029',
        '12:00',
        '13:30',
        'Miércoles'
    ),
    (
        'D_OPT_081',
        'G_OPT_029',
        '12:00',
        '13:30',
        'Viernes'
    ),
    (
        'D_OPT_082',
        'G_OPT_031',
        '07:00',
        '08:30',
        'Martes'
    ),
    (
        'D_OPT_083',
        'G_OPT_031',
        '07:00',
        '08:30',
        'Jueves'
    ),
    (
        'D_OPT_084',
        'G_OPT_031',
        '07:00',
        '08:30',
        'Viernes'
    );

-- IIA Semestre 6 - Vespertino
INSERT INTO
    distribucion (
        id,
        id_grupo,
        hora_ini,
        hora_fin,
        dia
    )
VALUES (
        'D_OPT_085',
        'G_OPT_026',
        '18:30',
        '20:00',
        'Lunes'
    ),
    (
        'D_OPT_086',
        'G_OPT_026',
        '18:30',
        '20:00',
        'Miércoles'
    ),
    (
        'D_OPT_087',
        'G_OPT_026',
        '18:30',
        '20:00',
        'Viernes'
    ),
    (
        'D_OPT_088',
        'G_OPT_028',
        '17:00',
        '18:30',
        'Lunes'
    ),
    (
        'D_OPT_089',
        'G_OPT_028',
        '17:00',
        '18:30',
        'Miércoles'
    ),
    (
        'D_OPT_090',
        'G_OPT_028',
        '17:00',
        '18:30',
        'Viernes'
    ),
    (
        'D_OPT_091',
        'G_OPT_030',
        '15:00',
        '16:30',
        'Lunes'
    ),
    (
        'D_OPT_092',
        'G_OPT_030',
        '15:00',
        '16:30',
        'Miércoles'
    ),
    (
        'D_OPT_093',
        'G_OPT_030',
        '15:00',
        '16:30',
        'Viernes'
    );

-- IIA Semestre 7 - Matutino
INSERT INTO
    distribucion (
        id,
        id_grupo,
        hora_ini,
        hora_fin,
        dia
    )
VALUES (
        'D_OPT_094',
        'G_OPT_033',
        '10:30',
        '12:00',
        'Lunes'
    ),
    (
        'D_OPT_095',
        'G_OPT_033',
        '10:30',
        '12:00',
        'Miércoles'
    ),
    (
        'D_OPT_096',
        'G_OPT_033',
        '10:30',
        '12:00',
        'Viernes'
    ),
    (
        'D_OPT_097',
        'G_OPT_035',
        '08:30',
        '10:00',
        'Lunes'
    ),
    (
        'D_OPT_098',
        'G_OPT_035',
        '08:30',
        '10:00',
        'Miércoles'
    ),
    (
        'D_OPT_099',
        'G_OPT_035',
        '08:30',
        '10:00',
        'Viernes'
    ),
    (
        'D_OPT_100',
        'G_OPT_037',
        '12:00',
        '13:30',
        'Lunes'
    ),
    (
        'D_OPT_101',
        'G_OPT_037',
        '12:00',
        '13:30',
        'Miércoles'
    ),
    (
        'D_OPT_102',
        'G_OPT_037',
        '12:00',
        '13:30',
        'Viernes'
    );

-- IIA Semestre 7 - Vespertino
INSERT INTO
    distribucion (
        id,
        id_grupo,
        hora_ini,
        hora_fin,
        dia
    )
VALUES (
        'D_OPT_103',
        'G_OPT_032',
        '18:30',
        '20:00',
        'Lunes'
    ),
    (
        'D_OPT_104',
        'G_OPT_032',
        '18:30',
        '20:00',
        'Miércoles'
    ),
    (
        'D_OPT_105',
        'G_OPT_032',
        '18:30',
        '20:00',
        'Viernes'
    ),
    (
        'D_OPT_106',
        'G_OPT_034',
        '17:00',
        '18:30',
        'Lunes'
    ),
    (
        'D_OPT_107',
        'G_OPT_034',
        '17:00',
        '18:30',
        'Miércoles'
    ),
    (
        'D_OPT_108',
        'G_OPT_034',
        '17:00',
        '18:30',
        'Viernes'
    ),
    (
        'D_OPT_109',
        'G_OPT_036',
        '15:00',
        '16:30',
        'Lunes'
    ),
    (
        'D_OPT_110',
        'G_OPT_036',
        '15:00',
        '16:30',
        'Miércoles'
    ),
    (
        'D_OPT_111',
        'G_OPT_036',
        '15:00',
        '16:30',
        'Viernes'
    ),
    (
        'D_OPT_112',
        'G_OPT_038',
        '20:00',
        '21:30',
        'Martes'
    ),
    (
        'D_OPT_113',
        'G_OPT_038',
        '20:00',
        '21:30',
        'Jueves'
    ),
    (
        'D_OPT_114',
        'G_OPT_038',
        '20:00',
        '21:30',
        'Viernes'
    );

-- LCD Semestre 6 - Matutino
INSERT INTO
    distribucion (
        id,
        id_grupo,
        hora_ini,
        hora_fin,
        dia
    )
VALUES (
        'D_OPT_115',
        'G_OPT_039',
        '10:30',
        '12:00',
        'Lunes'
    ),
    (
        'D_OPT_116',
        'G_OPT_039',
        '10:30',
        '12:00',
        'Miércoles'
    ),
    (
        'D_OPT_117',
        'G_OPT_039',
        '10:30',
        '12:00',
        'Viernes'
    ),
    (
        'D_OPT_118',
        'G_OPT_041',
        '08:30',
        '10:00',
        'Lunes'
    ),
    (
        'D_OPT_119',
        'G_OPT_041',
        '08:30',
        '10:00',
        'Miércoles'
    ),
    (
        'D_OPT_120',
        'G_OPT_041',
        '08:30',
        '10:00',
        'Viernes'
    ),
    (
        'D_OPT_121',
        'G_OPT_043',
        '12:00',
        '13:30',
        'Lunes'
    ),
    (
        'D_OPT_122',
        'G_OPT_043',
        '12:00',
        '13:30',
        'Miércoles'
    ),
    (
        'D_OPT_123',
        'G_OPT_043',
        '12:00',
        '13:30',
        'Viernes'
    );

-- LCD Semestre 6 - Vespertino
INSERT INTO
    distribucion (
        id,
        id_grupo,
        hora_ini,
        hora_fin,
        dia
    )
VALUES (
        'D_OPT_124',
        'G_OPT_040',
        '18:30',
        '20:00',
        'Lunes'
    ),
    (
        'D_OPT_125',
        'G_OPT_040',
        '18:30',
        '20:00',
        'Miércoles'
    ),
    (
        'D_OPT_126',
        'G_OPT_040',
        '18:30',
        '20:00',
        'Viernes'
    ),
    (
        'D_OPT_127',
        'G_OPT_042',
        '17:00',
        '18:30',
        'Lunes'
    ),
    (
        'D_OPT_128',
        'G_OPT_042',
        '17:00',
        '18:30',
        'Miércoles'
    ),
    (
        'D_OPT_129',
        'G_OPT_042',
        '17:00',
        '18:30',
        'Viernes'
    ),
    (
        'D_OPT_130',
        'G_OPT_044',
        '15:00',
        '16:30',
        'Lunes'
    ),
    (
        'D_OPT_131',
        'G_OPT_044',
        '15:00',
        '16:30',
        'Miércoles'
    ),
    (
        'D_OPT_132',
        'G_OPT_044',
        '15:00',
        '16:30',
        'Viernes'
    );

-- LCD Semestre 7 - Matutino
INSERT INTO
    distribucion (
        id,
        id_grupo,
        hora_ini,
        hora_fin,
        dia
    )
VALUES (
        'D_OPT_133',
        'G_OPT_045',
        '10:30',
        '12:00',
        'Lunes'
    ),
    (
        'D_OPT_134',
        'G_OPT_045',
        '10:30',
        '12:00',
        'Miércoles'
    ),
    (
        'D_OPT_135',
        'G_OPT_045',
        '10:30',
        '12:00',
        'Viernes'
    ),
    (
        'D_OPT_136',
        'G_OPT_047',
        '08:30',
        '10:00',
        'Lunes'
    ),
    (
        'D_OPT_137',
        'G_OPT_047',
        '08:30',
        '10:00',
        'Miércoles'
    ),
    (
        'D_OPT_138',
        'G_OPT_047',
        '08:30',
        '10:00',
        'Viernes'
    ),
    (
        'D_OPT_139',
        'G_OPT_049',
        '12:00',
        '13:30',
        'Lunes'
    ),
    (
        'D_OPT_140',
        'G_OPT_049',
        '12:00',
        '13:30',
        'Miércoles'
    ),
    (
        'D_OPT_141',
        'G_OPT_049',
        '12:00',
        '13:30',
        'Viernes'
    ),
    (
        'D_OPT_142',
        'G_OPT_051',
        '07:00',
        '08:30',
        'Martes'
    ),
    (
        'D_OPT_143',
        'G_OPT_051',
        '07:00',
        '08:30',
        'Jueves'
    ),
    (
        'D_OPT_144',
        'G_OPT_051',
        '07:00',
        '08:30',
        'Viernes'
    );

-- LCD Semestre 7 - Vespertino
INSERT INTO
    distribucion (
        id,
        id_grupo,
        hora_ini,
        hora_fin,
        dia
    )
VALUES (
        'D_OPT_145',
        'G_OPT_046',
        '18:30',
        '20:00',
        'Lunes'
    ),
    (
        'D_OPT_146',
        'G_OPT_046',
        '18:30',
        '20:00',
        'Miércoles'
    ),
    (
        'D_OPT_147',
        'G_OPT_046',
        '18:30',
        '20:00',
        'Viernes'
    ),
    (
        'D_OPT_148',
        'G_OPT_048',
        '17:00',
        '18:30',
        'Lunes'
    ),
    (
        'D_OPT_149',
        'G_OPT_048',
        '17:00',
        '18:30',
        'Miércoles'
    ),
    (
        'D_OPT_150',
        'G_OPT_048',
        '17:00',
        '18:30',
        'Viernes'
    ),
    (
        'D_OPT_151',
        'G_OPT_050',
        '15:00',
        '16:30',
        'Lunes'
    ),
    (
        'D_OPT_152',
        'G_OPT_050',
        '15:00',
        '16:30',
        'Miércoles'
    ),
    (
        'D_OPT_153',
        'G_OPT_050',
        '15:00',
        '16:30',
        'Viernes'
    );

-- ==================================================================
-- 8. OTROS DATOS DE PRUEBA NECESARIOS (Fechas y Chat originales)
-- ==================================================================
DELETE FROM fechas_relevantes;

INSERT INTO
    fechas_relevantes (
        inicio_semestre,
        fin_semestre,
        registro_primer_parcial,
        fin_registro_primer_parcial,
        registro_segundo_parcial,
        fin_registro_segundo_parcial,
        registro_tercer_parcial,
        fin_registro_tercer_parcial,
        registro_extra,
        fin_registro_extra,
        evalu_profe,
        fin_evalu_profe,
        inscribir_ets,
        fin_inscribir_ets,
        subir_doc_ets,
        fin_subir_doc_ets,
        eval_ets,
        fin_evalu_ets,
        periodo
    )
VALUES (
        '2025-02-03 00:00:00', -- inicio_semestre
        '2025-06-20 23:59:59', -- fin_semestre
        '2025-12-01 00:00:00', -- registro_primer_parcial
        '2025-12-07 23:59:59', -- fin_registro_primer_parcial
        '2025-04-01 00:00:00', -- registro_segundo_parcial
        '2025-04-07 23:59:59', -- fin_registro_segundo_parcial
        '2025-05-01 00:00:00', -- registro_tercer_parcial
        '2025-05-07 23:59:59', -- fin_registro_tercer_parcial
        '2025-06-10 00:00:00', -- registro_extra
        '2025-06-12 23:59:59', -- fin_registro_extra
        '2025-12-01 00:00:00', -- evalu_profe
        '2025-12-15 23:59:59', -- fin_evalu_profe
        '2025-12-01 00:00:00', -- inscribir_ets
        '2025-12-30 23:59:59', -- fin_inscribir_ets
        '2025-12-01 00:00:00', -- subir_doc_ets
        '2025-12-15 23:59:59', -- fin_subir_doc_ets
        '2025-12-01 00:00:00', -- eval_ets
        '2025-12-30 23:59:59', -- fin_evalu_ets
        '2025-1' -- periodo
    );

SELECT * FROM unidad_de_aprendizaje WHERE tipo = "OBLIGATORIA";

SELECT * FROM unidad_de_aprendizaje;

SELECT * FROM datos_personales;

delete from inscripcion;

INSERT INTO
    materia_reprobada (
        id,
        id_estudiante,
        id_ua,
        periodos_restantes,
        recurse,
        estado_actual
    )
VALUES (
        '457',
        'EST2021630309',
        'UA0015',
        '1',
        '0',
        'Reprobada'
    );

INSERT INTO
    materia_reprobada (
        id,
        id_estudiante,
        id_ua,
        periodos_restantes,
        recurse,
        estado_actual
    )
VALUES (
        '8910',
        'EST2021630309',
        'UA0016',
        '1',
        '0',
        'Reprobada'
    );

SELECT
    e.id AS id_estudiante,
    e.id_usuario,
    e.promedio,
    e.creditos_disponibles,
    e.estado_academico,
    i.id AS id_inscripcion,
    i.fecha_hora_in,
    i.fecha_hora_cad
FROM estudiante e
    INNER JOIN inscripcion i ON e.id_usuario = i.id_alumno
ORDER BY e.promedio DESC;

UPDATE inscripcion
set
    fecha_hora_in = "2025-11-24 18:00:00",
    fecha_hora_cad = "2025-11-24 22:00:00"
where
    id_alumno = "2021630310";

-- ==================================================================
-- ACTUALIZAR MAT_INSCRITOS CON CALIFICACIONES FINALES Y EXTRA
-- ==================================================================
UPDATE mat_inscritos
SET
    calificacion_primer = 8.0,
    calificacion_segundo = 8.0,
    calificacion_tercer = 8.0,
    extra = NULL,
    calificacion_final = 8.0
WHERE
    id = 'M0001';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.0,
    calificacion_segundo = 8.5,
    calificacion_tercer = 8.5,
    extra = NULL,
    calificacion_final = 8.3
WHERE
    id = 'M0002';

UPDATE mat_inscritos
SET
    calificacion_primer = 6.0,
    calificacion_segundo = 6.5,
    calificacion_tercer = 7.0,
    extra = NULL,
    calificacion_final = 7.5
WHERE
    id = 'M0003';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.5,
    calificacion_segundo = 9.0,
    calificacion_tercer = 9.0,
    extra = NULL,
    calificacion_final = 8.9
WHERE
    id = 'M0004';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.0,
    calificacion_segundo = 8.5,
    calificacion_tercer = 9.0,
    extra = NULL,
    calificacion_final = 8.5
WHERE
    id = 'M0005';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.0,
    calificacion_segundo = 8.0,
    calificacion_tercer = 8.5,
    extra = NULL,
    calificacion_final = 8.2
WHERE
    id = 'M0006';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.0,
    calificacion_segundo = 8.0,
    calificacion_tercer = 8.0,
    extra = NULL,
    calificacion_final = 8.0
WHERE
    id = 'M0007';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.5,
    calificacion_segundo = 8.5,
    calificacion_tercer = 9.0,
    extra = NULL,
    calificacion_final = 8.6
WHERE
    id = 'M0008';

UPDATE mat_inscritos
SET
    calificacion_primer = 7.0,
    calificacion_segundo = 7.5,
    calificacion_tercer = 8.0,
    extra = NULL,
    calificacion_final = 7.6
WHERE
    id = 'M0009';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.5,
    calificacion_segundo = 8.5,
    calificacion_tercer = 9.0,
    extra = NULL,
    calificacion_final = 8.8
WHERE
    id = 'M0010';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.0,
    calificacion_segundo = 8.0,
    calificacion_tercer = 8.5,
    extra = NULL,
    calificacion_final = 8.2
WHERE
    id = 'M0011';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.5,
    calificacion_segundo = 9.0,
    calificacion_tercer = 9.5,
    extra = NULL,
    calificacion_final = 9.0
WHERE
    id = 'M0012';

UPDATE mat_inscritos
SET
    calificacion_primer = 7.0,
    calificacion_segundo = 8.0,
    calificacion_tercer = 8.5,
    extra = NULL,
    calificacion_final = 7.9
WHERE
    id = 'M0013';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.0,
    calificacion_segundo = 8.5,
    calificacion_tercer = 9.0,
    extra = NULL,
    calificacion_final = 8.5
WHERE
    id = 'M0014';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.5,
    calificacion_segundo = 8.5,
    calificacion_tercer = 9.0,
    extra = NULL,
    calificacion_final = 8.8
WHERE
    id = 'M0015';

UPDATE mat_inscritos
SET
    calificacion_primer = 7.0,
    calificacion_segundo = 7.5,
    calificacion_tercer = 8.0,
    extra = NULL,
    calificacion_final = 7.4
WHERE
    id = 'M0016';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.0,
    calificacion_segundo = 8.5,
    calificacion_tercer = 9.0,
    extra = NULL,
    calificacion_final = 8.4
WHERE
    id = 'M0017';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.0,
    calificacion_segundo = 8.0,
    calificacion_tercer = 8.0,
    extra = NULL,
    calificacion_final = 8.0
WHERE
    id = 'M0018';

UPDATE mat_inscritos
SET
    calificacion_primer = 6.0,
    calificacion_segundo = 7.0,
    calificacion_tercer = 7.5,
    extra = NULL,
    calificacion_final = 6.9
WHERE
    id = 'M0019';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.0,
    calificacion_segundo = 8.0,
    calificacion_tercer = 8.5,
    extra = NULL,
    calificacion_final = 8.2
WHERE
    id = 'M0020';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.0,
    calificacion_segundo = 8.5,
    calificacion_tercer = 8.5,
    extra = NULL,
    calificacion_final = 8.3
WHERE
    id = 'M0021';

UPDATE mat_inscritos
SET
    calificacion_primer = 9.0,
    calificacion_segundo = 9.0,
    calificacion_tercer = 9.0,
    extra = NULL,
    calificacion_final = 9.0
WHERE
    id = 'M0022';

UPDATE mat_inscritos
SET
    calificacion_primer = 7.0,
    calificacion_segundo = 7.5,
    calificacion_tercer = 8.5,
    extra = NULL,
    calificacion_final = 7.7
WHERE
    id = 'M0023';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.5,
    calificacion_segundo = 8.5,
    calificacion_tercer = 9.0,
    extra = NULL,
    calificacion_final = 8.8
WHERE
    id = 'M0024';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.0,
    calificacion_segundo = 8.5,
    calificacion_tercer = 9.0,
    extra = NULL,
    calificacion_final = 8.5
WHERE
    id = 'M0025';

UPDATE mat_inscritos
SET
    calificacion_primer = 7.0,
    calificacion_segundo = 8.0,
    calificacion_tercer = 8.5,
    extra = NULL,
    calificacion_final = 7.9
WHERE
    id = 'M0026';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.0,
    calificacion_segundo = 8.5,
    calificacion_tercer = 8.5,
    extra = NULL,
    calificacion_final = 8.3
WHERE
    id = 'M0027';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.5,
    calificacion_segundo = 8.5,
    calificacion_tercer = 9.0,
    extra = NULL,
    calificacion_final = 8.6
WHERE
    id = 'M0028';

UPDATE mat_inscritos
SET
    calificacion_primer = 6.5,
    calificacion_segundo = 7.0,
    calificacion_tercer = 7.5,
    extra = NULL,
    calificacion_final = 7.5
WHERE
    id = 'M0029';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.5,
    calificacion_segundo = 9.0,
    calificacion_tercer = 9.0,
    extra = NULL,
    calificacion_final = 8.9
WHERE
    id = 'M0030';
-- =============================
-- Alumno 2023630107 (3BM1)
-- =============================

UPDATE mat_inscritos
SET
    calificacion_primer = 8.0,
    calificacion_segundo = 8.5,
    calificacion_tercer = 9.0,
    calificacion_final = 8.6,
    extra = NULL
WHERE
    id = 'M0031';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.0,
    calificacion_segundo = 8.0,
    calificacion_tercer = 9.0,
    calificacion_final = 8.2,
    extra = NULL
WHERE
    id = 'M0032';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.5,
    calificacion_segundo = 9.0,
    calificacion_tercer = 9.0,
    calificacion_final = 8.9,
    extra = NULL
WHERE
    id = 'M0033';

UPDATE mat_inscritos
SET
    calificacion_primer = 7.5,
    calificacion_segundo = 8.0,
    calificacion_tercer = 8.5,
    calificacion_final = 8.1,
    extra = NULL
WHERE
    id = 'M0034';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.5,
    calificacion_segundo = 8.5,
    calificacion_tercer = 9.0,
    calificacion_final = 8.7,
    extra = NULL
WHERE
    id = 'M0035';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.0,
    calificacion_segundo = 8.5,
    calificacion_tercer = 9.0,
    calificacion_final = 8.4,
    extra = NULL
WHERE
    id = 'M0036';

-- =============================
-- Alumno 2022630208 (5AM1)
-- =============================

UPDATE mat_inscritos
SET
    calificacion_primer = 8.0,
    calificacion_segundo = 8.0,
    calificacion_tercer = 8.5,
    calificacion_final = 8.1,
    extra = NULL
WHERE
    id = 'M0037';

UPDATE mat_inscritos
SET
    calificacion_primer = 9.0,
    calificacion_segundo = 9.0,
    calificacion_tercer = 9.5,
    calificacion_final = 9.2,
    extra = NULL
WHERE
    id = 'M0038';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.0,
    calificacion_segundo = 8.5,
    calificacion_tercer = 9.0,
    calificacion_final = 8.5,
    extra = NULL
WHERE
    id = 'M0039';

UPDATE mat_inscritos
SET
    calificacion_primer = 7.0,
    calificacion_segundo = 7.5,
    calificacion_tercer = 8.0,
    calificacion_final = 7.8,
    extra = NULL
WHERE
    id = 'M0040';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.5,
    calificacion_segundo = 8.5,
    calificacion_tercer = 9.0,
    calificacion_final = 8.7,
    extra = NULL
WHERE
    id = 'M0041';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.0,
    calificacion_segundo = 8.5,
    calificacion_tercer = 8.5,
    calificacion_final = 8.3,
    extra = NULL
WHERE
    id = 'M0042';

-- =============================
-- Alumno 2021630309 (7CM1)
-- =============================

UPDATE mat_inscritos
SET
    calificacion_primer = 8.0,
    calificacion_segundo = 8.5,
    calificacion_tercer = 9.0,
    calificacion_final = 8.5,
    extra = NULL
WHERE
    id = 'M0043';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.0,
    calificacion_segundo = 8.0,
    calificacion_tercer = 8.5,
    calificacion_final = 8.2,
    extra = NULL
WHERE
    id = 'M0044';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.5,
    calificacion_segundo = 9.0,
    calificacion_tercer = 9.0,
    calificacion_final = 8.8,
    extra = NULL
WHERE
    id = 'M0045';

-- =============================
-- Alumno 2021630310 (8CM1)
-- =============================

UPDATE mat_inscritos
SET
    calificacion_primer = 8.5,
    calificacion_segundo = 8.5,
    calificacion_tercer = 9.0,
    calificacion_final = 8.7,
    extra = NULL
WHERE
    id = 'M0046';

UPDATE mat_inscritos
SET
    calificacion_primer = 9.0,
    calificacion_segundo = 9.0,
    calificacion_tercer = 9.5,
    calificacion_final = 9.1,
    extra = NULL
WHERE
    id = 'M0047';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.0,
    calificacion_segundo = 8.5,
    calificacion_tercer = 9.0,
    calificacion_final = 8.4,
    extra = NULL
WHERE
    id = 'M0048';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.5,
    calificacion_segundo = 9.0,
    calificacion_tercer = 9.0,
    calificacion_final = 8.9,
    extra = NULL
WHERE
    id = 'M0049';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.0,
    calificacion_segundo = 8.5,
    calificacion_tercer = 9.0,
    calificacion_final = 8.5,
    extra = NULL
WHERE
    id = 'M0050';

UPDATE mat_inscritos
SET
    calificacion_primer = 8.5,
    calificacion_segundo = 8.5,
    calificacion_tercer = 9.5,
    calificacion_final = 8.8,
    extra = NULL
WHERE
    id = 'M0051';

-- Alumno 2021630310 - Sin reprobaciones (alumno excelente)
-- Se mantienen todas sus calificaciones sin cambios

-- Verificar las actualizaciones
SELECT
    g.nombre AS grupo,
    ua.nombre AS materia,
    mi.calificacion_primer,
    mi.calificacion_segundo,
    mi.calificacion_tercer,
    mi.calificacion_final,
    mi.extra,
    k.creditos_obtenidos,
    e.creditos_disponibles
FROM
    mat_inscritos mi
    INNER JOIN horario h ON mi.id_horario = h.id
    INNER JOIN grupo g ON mi.id_grupo = g.id
    INNER JOIN unidad_de_aprendizaje ua ON g.id_ua = ua.id
    INNER JOIN kardex k ON k.id_alumno = h.id_alumno
    INNER JOIN estudiante e ON e.id_usuario = h.id_alumno
ORDER BY h.id_alumno, g.nombre;

SELECT h.id, k.promedio, k.situacion_academica, e.promedio, e.estado_academico
FROM
    datos_personales h
    INNER JOIN kardex k ON k.id_alumno = h.id
    INNER JOIN estudiante e ON e.id_usuario = h.id
ORDER BY k.promedio;