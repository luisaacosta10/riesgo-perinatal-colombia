"""Carga de los modelos finales guardados en modelos/.

El modelo de bajo peso (regresión logística) usa una clase propia, OneHotRefModa,
que en el artículo se definió dentro del .qmd (es decir, en __main__). joblib la busca
ahí al cargar, así que antes de cargar se registra en __main__. Úsese siempre
cargar_modelo() en lugar de joblib.load() directo.
"""
import json
from pathlib import Path

import __main__
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

RAIZ = Path(__file__).resolve().parent


class OneHotRefModa(OneHotEncoder):
    """One-hot que toma como referencia la categoría más frecuente del
    entrenamiento (como la logística de la E2), calculada en fit."""

    def fit(self, X, y=None):
        Xd = pd.DataFrame(X)
        self.drop = [Xd[c].value_counts().index[0] for c in Xd.columns]
        return super().fit(X, y)


# Misma limpieza que el artículo: códigos de "sin información" que pasan a nulos.
CODIGOS_SIN_INFO = {
    "NUMCONSUL": ["99"], "EDAD_MADRE": ["99"], "N_EMB": ["99"], "N_HIJOSV": ["99"],
    "EDAD_PADRE": ["999"], "NIV_EDUM": ["99"], "NIV_EDUP": ["99"], "EST_CIVM": ["9"],
    "SEG_SOCIAL": ["9"], "MUL_PARTO": ["9"], "IDPERTET": ["9"],
    "ULTCURMAD": ["99"], "ULTCURPAD": ["99"],
}
NUMERICAS = ["NUMCONSUL", "N_EMB", "N_HIJOSV", "EDAD_PADRE"]


def leer_base(ruta="BD-EEVV-Nacimientos-2023.csv", **kw):
    """Lee la base del DANE como texto (conserva los ceros a la izquierda de los códigos)."""
    return pd.read_csv(ruta, encoding="cp1252", dtype=str, **kw)


def preparar_datos(df):
    """Limpia registros con las columnas del DANE, en texto, igual que en el artículo.
    Los modelos reciben el resultado y devuelven la predicción sin más pasos."""
    d = df.copy()
    for var, codigos in CODIGOS_SIN_INFO.items():
        if var in d.columns:
            col = d[var].astype(object).where(d[var].notna(), np.nan)
            d[var] = col.mask(col.isin(codigos))      # nulos como np.nan (no None): así los imputa sklearn
    for var in NUMERICAS:
        if var in d.columns:
            d[var] = pd.to_numeric(d[var], errors="coerce")
    return d


def cargar_modelo(ruta):
    """Carga un .joblib de modelos/ (registra antes la clase propia)."""
    if not hasattr(__main__, "OneHotRefModa"):
        __main__.OneHotRefModa = OneHotRefModa
    return joblib.load(ruta)


def cargar_meta():
    """Columnas, umbral y tipo de modelo de cada objetivo (modelos/modelos_e4.json)."""
    return json.loads((RAIZ / "modelos" / "modelos_e4.json").read_text(encoding="utf-8"))
