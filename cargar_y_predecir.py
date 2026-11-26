#!/usr/bin/env python3
"""
cargar_y_predecir.py
====================
Predicción de cesárea, bajo peso al nacer y tiempo de gestación a partir de
registros de nacimientos (DANE, Estadísticas Vitales 2023).

Carga los tres pipelines de scikit-learn guardados en ``modelos/`` (.joblib),
lee un archivo con registros nuevos y escribe un CSV con:

* la probabilidad de cesárea y de bajo peso, y la clase según el umbral
  fijado en la E3 (cesárea 0.50, bajo peso 0.16);
* las semanas de gestación estimadas por la regresión lineal múltiple.

Los modelos NO se reentrenan: se usan tal como se evaluaron en la prueba.

Uso rápido
----------
    # 1) Generar una plantilla con las columnas que se esperan
    python cargar_y_predecir.py --plantilla plantilla_registros.csv

    # 2) Predecir
    python cargar_y_predecir.py -i registros.csv -o predicciones.csv

    # 3) Solo algunos objetivos
    python cargar_y_predecir.py -i registros.csv --objetivos cesarea bajo_peso

El archivo de entrada puede ser CSV (coma o punto y coma) o Excel, y puede
traer las 39 columnas originales del DANE: solo se usan las 16 predictoras
previas al parto. Los códigos de "sin información" del DANE (9, 99, 999) se
convierten en nulos antes de predecir, igual que en el entrenamiento.

Este módulo también lo importa ``app.py`` (aplicación Streamlit).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
DIR_MODELOS = RAIZ / "modelos"
ARCHIVO_CONFIG = "modelos_e4.json"

# --------------------------------------------------------------------------- #
# Variables y objetivos
# --------------------------------------------------------------------------- #

NUMERICAS = ["NUMCONSUL", "N_HIJOSV", "N_EMB", "EDAD_PADRE"]
CATEGORICAS = [
    "SEXO", "MUL_PARTO", "IDPERTET", "EDAD_MADRE", "EST_CIVM", "NIV_EDUM",
    "ULTCURMAD", "CODPTORE", "AREA_RES", "SEG_SOCIAL", "NIV_EDUP", "ULTCURPAD",
]
# Las 16 predictoras previas al parto (sin T_GES ni MES).
PREDICTORAS = NUMERICAS + CATEGORICAS

# Código con que el DANE marca "sin información" en cada variable (Tabla 16).
CODIGO_DESCONOCIDO = {
    "EST_CIVM": "9", "SEG_SOCIAL": "9", "MUL_PARTO": "9", "IDPERTET": "9",
    "NUMCONSUL": "99", "EDAD_MADRE": "99", "N_EMB": "99", "N_HIJOSV": "99",
    "NIV_EDUM": "99", "NIV_EDUP": "99", "ULTCURMAD": "99", "ULTCURPAD": "99",
    "EDAD_PADRE": "999",
}


@dataclass(frozen=True)
class Objetivo:
    clave: str
    nombre: str
    archivo: str
    tipo: str                      # "clasificacion" | "regresion"
    umbral_defecto: float | None
    clase_positiva: str = ""
    clase_negativa: str = ""
    alias: tuple[str, ...] = ()    # para buscar el bloque en modelos_e4.json


OBJETIVOS: dict[str, Objetivo] = {
    "cesarea": Objetivo(
        "cesarea", "Cesárea", "modelo_cesarea.joblib", "clasificacion", 0.50,
        "Cesárea", "No cesárea", ("cesarea",)),
    "bajo_peso": Objetivo(
        "bajo_peso", "Bajo peso al nacer", "modelo_bajo_peso.joblib",
        "clasificacion", 0.16, "Bajo peso", "Peso normal", ("bajo_peso",)),
    "tiempo_gestacion": Objetivo(
        "tiempo_gestacion", "Tiempo de gestación",
        "modelo_tiempo_gestacion.joblib", "regresion", None,
        alias=("tiempo_gestacion", "t_ges", "gestacion")),
}

# --------------------------------------------------------------------------- #
# Etiquetas legibles (diccionario DANE). Se usan en la app y como respaldo
# cuando no se pueden leer las categorías del pipeline.
# --------------------------------------------------------------------------- #

_EDUCACION = {
    "1": "Preescolar", "2": "Básica primaria", "3": "Básica secundaria",
    "4": "Media académica o clásica", "5": "Media técnica", "6": "Normalista",
    "7": "Técnica profesional", "8": "Tecnológica", "9": "Profesional",
    "10": "Especialización", "11": "Maestría", "12": "Doctorado",
    "13": "Ninguno",
}
_DEPARTAMENTOS = {
    "5": "Antioquia", "8": "Atlántico", "11": "Bogotá D.C.", "13": "Bolívar",
    "15": "Boyacá", "17": "Caldas", "18": "Caquetá", "19": "Cauca",
    "20": "Cesar", "23": "Córdoba", "25": "Cundinamarca", "27": "Chocó",
    "41": "Huila", "44": "La Guajira", "47": "Magdalena", "50": "Meta",
    "52": "Nariño", "54": "Norte de Santander", "63": "Quindío",
    "66": "Risaralda", "68": "Santander", "70": "Sucre", "73": "Tolima",
    "76": "Valle del Cauca", "81": "Arauca", "85": "Casanare",
    "86": "Putumayo", "88": "San Andrés y Providencia", "91": "Amazonas",
    "94": "Guainía", "95": "Guaviare", "97": "Vaupés", "99": "Vichada",
}
# Año o grado aprobado dentro del nivel educativo (el DANE no publica una tabla de
# categorías para esta variable): 0 = ninguno; 1-5 primaria, 6-9 secundaria, 10-11 media.
_ULTIMO_CURSO = {"0": "Ninguno"} | {str(i): f"Año o grado {i}" for i in range(1, 14)}
ETIQUETAS: dict[str, dict[str, str]] = {
    "SEXO": {"1": "Masculino", "2": "Femenino", "3": "Indeterminado"},
    "MUL_PARTO": {"1": "Simple", "2": "Doble", "3": "Triple",
                  "4": "Cuádruple o más"},
    "IDPERTET": {"1": "Indígena", "2": "Rom (gitano)", "3": "Raizal",
                 "4": "Palenquero", "5": "Negro, mulato o afrocolombiano",
                 "6": "Ninguno de los anteriores"},
    "EDAD_MADRE": {"1": "10 a 14 años", "2": "15 a 19 años",
                   "3": "20 a 24 años", "4": "25 a 29 años",
                   "5": "30 a 34 años", "6": "35 a 39 años",
                   "7": "40 a 44 años", "8": "45 a 49 años",
                   "9": "50 a 54 años"},
    "EST_CIVM": {"1": "Unión libre desde hace 2 años o más",
                 "2": "Unión libre hace menos de 2 años",
                 "3": "Separada o divorciada", "4": "Viuda",
                 "5": "Soltera", "6": "Casada"},
    "NIV_EDUM": _EDUCACION,
    "NIV_EDUP": _EDUCACION,
    "AREA_RES": {"1": "Cabecera municipal", "2": "Centro poblado",
                 "3": "Rural disperso"},
    "SEG_SOCIAL": {"1": "Contributivo", "2": "Subsidiado", "3": "Excepción",
                   "4": "Especial", "5": "No asegurado"},
    "CODPTORE": _DEPARTAMENTOS,
    "ULTCURMAD": _ULTIMO_CURSO,
    "ULTCURPAD": _ULTIMO_CURSO,
}
# Respaldo si el pipeline no permite leer sus categorías.
_CODIGOS_RESPALDO = {c: [str(i) for i in range(1, 14)]
                     for c in ("ULTCURMAD", "ULTCURPAD")}


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #

def normalizar_codigo(valor: Any) -> Any:
    """'04' -> '4', 3.0 -> '3', vacío / 'sin_info' -> NaN."""
    if valor is None or (not isinstance(valor, str) and pd.isna(valor)):
        return np.nan
    s = str(valor).strip()
    if s == "" or s.lower() in {"nan", "none", "na", "sin_info"}:
        return np.nan
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def etiqueta(columna: str, codigo: str) -> str:
    return ETIQUETAS.get(columna, {}).get(str(codigo), f"Código {codigo}")


def _sin_acentos(texto: str) -> str:
    t = unicodedata.normalize("NFD", texto)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return t.lower().replace(" ", "_")


def _clave_orden(k: str) -> tuple[int, Any]:
    return (0, int(k)) if str(k).isdigit() else (1, str(k))


def _buscar_bloque(nodo: Any, alias: Iterable[str]) -> dict:
    """Busca en el JSON el bloque de un modelo (tolera distintas estructuras)."""
    alias = tuple(alias)
    if isinstance(nodo, dict):
        for k, v in nodo.items():
            if isinstance(v, dict) and any(a in _sin_acentos(str(k)) for a in alias):
                return v
        for v in nodo.values():
            r = _buscar_bloque(v, alias)
            if r:
                return r
    elif isinstance(nodo, list):
        for v in nodo:
            if isinstance(v, dict):
                textos = " ".join(_sin_acentos(x) for x in v.values()
                                  if isinstance(x, str))
                if any(a in textos for a in alias):
                    return v
    return {}


def _importar_utils() -> None:
    """Hace disponibles las clases de utils_modelos.py para abrir los .joblib.

    Los pipelines se guardaron desde el notebook, así que joblib busca las
    clases propias (p. ej. OneHotRefModa) en ``__main__``. Se registran allí.
    """
    import inspect

    if str(RAIZ) not in sys.path:
        sys.path.insert(0, str(RAIZ))
    try:
        import utils_modelos
    except ImportError:
        return
    import __main__

    for nombre, obj in vars(utils_modelos).items():
        if nombre.startswith("_"):
            continue
        if (inspect.isclass(obj) or inspect.isfunction(obj)) and \
                not hasattr(__main__, nombre):
            setattr(__main__, nombre, obj)


def _extraer_categorias(pipe: Any) -> dict[str, dict[str, Any]]:
    """Lee del pipeline las categorías que vio cada columna en el entrenamiento.

    Devuelve {columna: {código_normalizado: categoría_original}}.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    def buscar_ohe(obj: Any):
        if isinstance(obj, OneHotEncoder):
            return obj
        for _, paso in getattr(obj, "steps", []):
            r = buscar_ohe(paso)
            if r is not None:
                return r
        return None

    resultado: dict[str, dict[str, Any]] = {}

    def recorrer(obj: Any) -> None:
        if isinstance(obj, ColumnTransformer):
            for _, trans, cols in getattr(obj, "transformers_", []):
                if isinstance(trans, str):
                    continue
                ohe = buscar_ohe(trans)
                if ohe is None or not hasattr(ohe, "categories_"):
                    continue
                cols = [cols] if isinstance(cols, str) else list(cols)
                for col, cats in zip(cols, ohe.categories_):
                    mapa = {}
                    for cat in cats:
                        n = normalizar_codigo(cat)
                        if isinstance(n, str):
                            mapa[n] = cat
                    resultado[str(col)] = mapa
        for _, paso in getattr(obj, "steps", []):
            recorrer(paso)

    try:
        recorrer(pipe)
    except Exception:  # la lectura es opcional: hay respaldo
        return {}
    return resultado


# --------------------------------------------------------------------------- #
# Lectura de archivos
# --------------------------------------------------------------------------- #

def leer_registros(fuente: Any, nombre: str | None = None) -> pd.DataFrame:
    """Lee CSV / Excel desde una ruta o un archivo subido (todo como texto).

    Se lee como texto para conservar códigos como '04' y evitar que pandas
    convierta columnas categóricas en decimales.
    """
    if isinstance(fuente, (str, Path)):
        ruta = Path(fuente)
        if not ruta.exists():
            raise FileNotFoundError(f"No existe el archivo de entrada: {ruta}")
        datos, nombre = ruta.read_bytes(), ruta.name
    else:
        datos = fuente.read() if hasattr(fuente, "read") else bytes(fuente)
        nombre = nombre or getattr(fuente, "name", "archivo.csv")

    sufijo = Path(nombre).suffix.lower()
    if sufijo in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(datos), dtype=str)
    for codificacion in ("utf-8-sig", "latin-1"):
        try:
            texto = datos.decode(codificacion)
            break
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.StringIO(texto), sep=None, engine="python", dtype=str)


def plantilla_registros() -> pd.DataFrame:
    """Dos filas de ejemplo con las 16 columnas que espera el modelo."""
    filas = [
        dict(NUMCONSUL=7, N_HIJOSV=1, N_EMB=1, EDAD_PADRE=28, SEXO=1,
             MUL_PARTO=1, IDPERTET=6, EDAD_MADRE=3, EST_CIVM=1, NIV_EDUM=4,
             ULTCURMAD=99, CODPTORE=13, AREA_RES=1, SEG_SOCIAL=2, NIV_EDUP=4,
             ULTCURPAD=99),
        dict(NUMCONSUL=3, N_HIJOSV=2, N_EMB=3, EDAD_PADRE=999, SEXO=2,
             MUL_PARTO=1, IDPERTET=6, EDAD_MADRE=2, EST_CIVM=5, NIV_EDUM=3,
             ULTCURMAD=99, CODPTORE=8, AREA_RES=3, SEG_SOCIAL=5, NIV_EDUP=99,
             ULTCURPAD=99),
    ]
    return pd.DataFrame(filas, columns=PREDICTORAS)


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #

class Predictor:
    """Carga los pipelines .joblib y genera las predicciones."""

    def __init__(self, dir_modelos: str | Path = DIR_MODELOS) -> None:
        self.dir = Path(dir_modelos)
        self.config = self._leer_config()
        self._pipes: dict[str, Any] = {}
        self._contratos: dict[str, dict] = {}
        self._categorias: dict[str, dict[str, Any]] | None = None
        _importar_utils()

    # -- configuración ------------------------------------------------------ #
    def _leer_config(self) -> dict:
        ruta = self.dir / ARCHIVO_CONFIG
        if not ruta.exists():
            return {}
        try:
            return json.loads(ruta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            warnings.warn(f"No se pudo leer {ruta}; se usan valores por defecto.")
            return {}

    def _bloque(self, clave: str) -> dict:
        return _buscar_bloque(self.config, OBJETIVOS[clave].alias)

    def contrato(self, clave: str) -> dict:
        """Diccionario guardado en el .joblib (vacío si es un pipeline suelto)."""
        self.pipeline(clave)
        return self._contratos.get(clave, {})

    def columnas(self, clave: str) -> list[str]:
        c = self.contrato(clave).get("columnas") if clave in self.disponibles else None
        if isinstance(c, list) and c:
            return [str(x) for x in c]
        bloque = self._bloque(clave)
        for k in ("columnas", "features", "predictores", "variables"):
            v = bloque.get(k)
            if isinstance(v, list) and v:
                return [str(c) for c in v]
        return list(PREDICTORAS)

    def umbral(self, clave: str) -> float | None:
        obj = OBJETIVOS[clave]
        if obj.tipo != "clasificacion":
            return None
        u = self.contrato(clave).get("umbral") if clave in self.disponibles else None
        if isinstance(u, (int, float)) and not isinstance(u, bool):
            return float(u)
        bloque = self._bloque(clave)
        for k in ("umbral", "u", "threshold", "umbral_decision"):
            v = bloque.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        return obj.umbral_defecto

    # -- modelos ------------------------------------------------------------ #
    @property
    def disponibles(self) -> list[str]:
        return [k for k, o in OBJETIVOS.items() if (self.dir / o.archivo).exists()]

    def pipeline(self, clave: str) -> Any:
        if clave not in self._pipes:
            ruta = self.dir / OBJETIVOS[clave].archivo
            if not ruta.exists():
                raise FileNotFoundError(
                    f"Falta {ruta}. Genera el modelo desde el .qmd (bloque de "
                    f"guardado de la E4) y vuelve a intentar.")
            try:
                m = joblib.load(ruta)
                if isinstance(m, dict) and "pipeline" in m:
                    self._contratos[clave] = m
                    m = m["pipeline"]
                self._pipes[clave] = m
            except AttributeError as e:
                raise RuntimeError(
                    f"No se pudo abrir {ruta.name}: el pipeline usa una clase que "
                    f"no está en utils_modelos.py ({e}). Revisa que utils_modelos.py "
                    f"esté en la misma carpeta y defina esa clase.") from e
        return self._pipes[clave]

    def cargar_todos(self) -> dict[str, str]:
        """Carga todos los modelos disponibles. Devuelve {clave: error}."""
        errores = {}
        for clave in self.disponibles:
            try:
                self.pipeline(clave)
            except Exception as e:  # noqa: BLE001
                errores[clave] = f"{type(e).__name__}: {e}"
        return errores

    def estado(self) -> list[dict[str, Any]]:
        return [dict(clave=k, nombre=o.nombre, archivo=o.archivo,
                     disponible=(self.dir / o.archivo).exists(),
                     umbral=self.umbral(k)) for k, o in OBJETIVOS.items()]

    # -- categorías --------------------------------------------------------- #
    def categorias(self) -> dict[str, dict[str, Any]]:
        if self._categorias is None:
            self._categorias = {}
            for clave in self.disponibles:
                try:
                    self._categorias.update(_extraer_categorias(self.pipeline(clave)))
                except Exception:  # noqa: BLE001
                    continue
        return self._categorias

    def categorias_leidas_del_modelo(self) -> bool:
        return all(c in self.categorias() for c in ("CODPTORE", "SEXO"))

    def opciones(self, columna: str, ordenar_por_etiqueta: bool = False
                 ) -> list[tuple[str, str]]:
        """Opciones [(código, etiqueta)] para un selector de la app."""
        cats = self.categorias().get(columna)
        if cats:
            codigos = sorted(cats, key=_clave_orden)
        elif columna in ETIQUETAS:
            codigos = sorted(ETIQUETAS[columna], key=_clave_orden)
        else:
            codigos = _CODIGOS_RESPALDO.get(columna, [])
        pares = [(c, etiqueta(columna, c)) for c in codigos]
        if ordenar_por_etiqueta:
            pares.sort(key=lambda p: p[1])
        return pares

    # -- preparación -------------------------------------------------------- #
    def _preparar(self, df: pd.DataFrame, columnas: list[str],
                  limpiar_codigos: bool, avisos: list[str]) -> pd.DataFrame:
        faltan = [c for c in columnas if c not in df.columns]
        if faltan:
            raise ValueError(
                "Faltan columnas en el archivo: " + ", ".join(faltan) +
                ". Usa --plantilla (o el botón de la app) para ver el formato.")
        cats = self.categorias()
        X = pd.DataFrame(index=df.index)
        for c in columnas:
            if c in NUMERICAS:
                v = pd.to_numeric(df[c], errors="coerce").astype(float)
                if limpiar_codigos and c in CODIGO_DESCONOCIDO:
                    v = v.mask(v == float(CODIGO_DESCONOCIDO[c]))
                X[c] = v
                continue
            v = df[c].map(normalizar_codigo)
            if limpiar_codigos and c in CODIGO_DESCONOCIDO:
                v = v.mask(v == CODIGO_DESCONOCIDO[c])
            mapa = cats.get(c)
            if mapa:
                nuevos = sorted({z for z in v.dropna().unique() if z not in mapa})
                if nuevos:
                    avisos.append(
                        f"{c}: códigos no vistos en el entrenamiento "
                        f"({', '.join(map(str, nuevos[:5]))}); se tratan como "
                        f"categoría desconocida.")
                v = v.map(lambda z: mapa.get(z, z) if isinstance(z, str) else z)
            X[c] = v.astype(object)
        return X

    # -- predicción --------------------------------------------------------- #
    def _llamar(self, clave: str, X: pd.DataFrame, metodo: str) -> np.ndarray:
        pipe = self.pipeline(clave)
        alterno = X.copy()
        for c in alterno.columns:
            if c not in NUMERICAS:
                alterno[c] = alterno[c].map(
                    lambda z: z if (not isinstance(z, str) and pd.isna(z)) else str(z)
                ).astype(object)
        ultimo: Exception | None = None
        for intento in (X, alterno):
            try:
                return getattr(pipe, metodo)(intento)
            except Exception as e:  # noqa: BLE001
                ultimo = ultimo or e
        raise RuntimeError(
            f"El pipeline '{clave}' no pudo predecir con las columnas recibidas. "
            f"Detalle: {type(ultimo).__name__}: {ultimo}") from ultimo

    def predecir(self, df: pd.DataFrame, objetivos: Iterable[str] | None = None,
                 limpiar_codigos: bool = True) -> pd.DataFrame:
        """Devuelve un DataFrame con probabilidades, clases y semanas."""
        claves = list(objetivos) if objetivos else self.disponibles
        if not claves:
            raise FileNotFoundError(
                f"No se encontraron modelos .joblib en {self.dir}.")
        avisos: list[str] = []
        salida = pd.DataFrame(index=df.index)
        umbrales: dict[str, float] = {}

        for clave in claves:
            obj = OBJETIVOS[clave]
            X = self._preparar(df, self.columnas(clave), limpiar_codigos, avisos)
            if obj.tipo == "clasificacion":
                pipe = self.pipeline(clave)
                clases = list(pipe.classes_)
                idx = clases.index(1) if 1 in clases else len(clases) - 1
                p = self._llamar(clave, X, "predict_proba")[:, idx]
                u = self.umbral(clave)
                umbrales[clave] = u
                salida[f"prob_{clave}"] = np.round(p, 4)
                salida[f"clase_{clave}"] = np.where(
                    p >= u, obj.clase_positiva, obj.clase_negativa)
            else:
                semanas = self._llamar(clave, X, "predict")
                salida["semanas_gestacion"] = np.round(np.asarray(semanas, float), 1)

        salida.attrs["avisos"] = sorted(set(avisos))
        salida.attrs["umbrales"] = umbrales
        return salida


# --------------------------------------------------------------------------- #
# Línea de comandos
# --------------------------------------------------------------------------- #

def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cargar_y_predecir.py",
        description="Predice cesárea, bajo peso y tiempo de gestación con los "
                    "modelos guardados (.joblib) de la E3/E4.")
    p.add_argument("-i", "--entrada", help="CSV o Excel con los registros.")
    p.add_argument("-o", "--salida", help="CSV de salida "
                   "(por defecto: predicciones_<entrada>.csv).")
    p.add_argument("-m", "--modelos", default=str(DIR_MODELOS),
                   help="Carpeta con los .joblib (por defecto: modelos/).")
    p.add_argument("--objetivos", nargs="+", choices=list(OBJETIVOS),
                   help="Objetivos a predecir (por defecto: todos).")
    p.add_argument("--col-id", help="Columna del archivo que identifica cada "
                   "registro; se copia a la salida.")
    p.add_argument("--incluir-entrada", action="store_true",
                   help="Añade las 16 predictoras a la salida.")
    p.add_argument("--sin-limpieza", action="store_true",
                   help="No convertir 9/99/999 en 'sin información'.")
    p.add_argument("--plantilla", metavar="RUTA",
                   help="Escribe un CSV de ejemplo con las columnas esperadas y termina.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.plantilla:
        plantilla_registros().to_csv(args.plantilla, index=False)
        print(f"Plantilla escrita en: {args.plantilla}")
        return 0
    if not args.entrada:
        print("Error: indica el archivo de entrada con -i/--entrada "
              "(o usa --plantilla).", file=sys.stderr)
        return 1

    try:
        df = leer_registros(args.entrada)
        predictor = Predictor(args.modelos)
        resultado = predictor.predecir(
            df, args.objetivos, limpiar_codigos=not args.sin_limpieza)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    umbrales = resultado.attrs.get("umbrales", {})
    avisos = resultado.attrs.get("avisos", [])
    if args.col_id and args.col_id in df.columns:
        resultado.insert(0, args.col_id, df[args.col_id].values)
    else:
        resultado.insert(0, "fila", np.arange(1, len(df) + 1))
    if args.incluir_entrada:
        resultado = pd.concat([resultado, df[PREDICTORAS]], axis=1)

    salida = Path(args.salida) if args.salida else \
        Path(args.entrada).with_name(f"predicciones_{Path(args.entrada).stem}.csv")
    resultado.to_csv(salida, index=False, encoding="utf-8-sig")

    # Resumen en consola
    print(f"\nRegistros procesados: {_fmt(len(df))}")
    for clave, u in umbrales.items():
        marcados = (resultado[f"clase_{clave}"] == OBJETIVOS[clave].clase_positiva).mean()
        print(f"  {OBJETIVOS[clave].nombre:<20} umbral {u:.2f} | "
              f"marcados {marcados:6.1%} | prob. media "
              f"{resultado[f'prob_{clave}'].mean():.3f}")
    if "semanas_gestacion" in resultado:
        print(f"  {'Tiempo de gestación':<20} media estimada "
              f"{resultado['semanas_gestacion'].mean():.1f} semanas "
              f"(aproximación, R² ≈ 0.07)")
    for aviso in avisos:
        print(f"  Aviso: {aviso}")
    print(f"\nResultados guardados en: {salida}")
    print("Recordatorio: son alertas de apoyo, no decisiones sobre una "
          "gestante en particular.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
