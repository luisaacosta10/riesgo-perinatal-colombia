"""
app.py
======
Aplicación interactiva (Streamlit) para explorar los tres modelos de la E4:

* probabilidad de cesárea (Random Forest, umbral 0.50),
* probabilidad de bajo peso al nacer (regresión logística, umbral 0.16),
* semanas de gestación estimadas (regresión lineal múltiple).

Usa exactamente los pipelines .joblib y los umbrales de la E3, sin reentrenar.

Ejecución:
    pip install streamlit
    streamlit run app.py

La lógica de carga y predicción vive en ``cargar_y_predecir.py``.
"""

from __future__ import annotations

import base64
import math

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from cargar_y_predecir import (
    DIR_MODELOS, NUMERICAS, OBJETIVOS, PREDICTORAS, Predictor, leer_registros,
    plantilla_registros,
)

# Frecuencia de cada desenlace en la base de trabajo (artículo, Sección 1.3).
PREVALENCIA = {"cesarea": 0.502, "bajo_peso": 0.110}
SIN_INFO = "__sin_info__"
RECM_SEMANAS = 3.09  # error típico de la regresión (artículo, Tabla 34)

# Paleta "Marea": azul petróleo profundo, verde agua, dorado y baya.
TINTA, AGUA, AGUA_CLARA = "#0E2F3A", "#1F7A8C", "#8FD3DC"
DORADO, BAYA, PINO = "#D9A441", "#B8465A", "#2E8B6E"

st.set_page_config(page_title="Riesgo perinatal · DANE 2023", page_icon="🩺",
                   layout="wide", initial_sidebar_state="expanded")

# =========================================================================== #
# ESTILO
# =========================================================================== #
ESTILO = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');
:root{
  --tinta:#0E2F3A; --agua:#1F7A8C; --agua-2:#5FB3BF; --bruma:#EAF3F4;
  --papel:#F6F9F9; --linea:#D5E3E6; --tenue:#5B7480;
  --dorado:#D9A441; --baya:#B8465A; --pino:#2E8B6E;
  --serif:'Fraunces',Georgia,'Times New Roman',serif;
  --sans:'DM Sans',system-ui,-apple-system,'Segoe UI',sans-serif;
}
html{color-scheme:light;}
.stApp{background:var(--papel);color:var(--tinta);font-family:var(--sans);}
.stApp :where(p,li,label,input,textarea,button,td,th,small,b,em,h1,h2,h3,h4,h5){font-family:var(--sans);}
[data-testid="stHeader"]{background:transparent;}
#MainMenu,footer{visibility:hidden;}
.stAppDeployButton,[data-testid="stAppDeployButton"],[data-testid="stDeployButton"]{display:none!important;}
.block-container{padding-top:1.6rem;padding-bottom:3rem;max-width:1180px;}

/* ---------- portada ---------- */
.hero{background:var(--tinta);border-radius:22px;padding:2.6rem 2.8rem;display:grid;
  grid-template-columns:1.2fr 1fr;align-items:center;gap:1rem;overflow:hidden;
  margin-bottom:1.1rem;}
.hero h1{font-family:var(--serif);font-weight:600;font-size:2.7rem;line-height:1.08;
  color:#F2F8F9;margin:0 0 .8rem 0;padding:0;letter-spacing:-.01em;}
.hero p{color:#B9D3D8;font-size:1.05rem;line-height:1.55;max-width:46ch;margin:0 0 1.3rem 0;}
.hero .datos{display:flex;flex-wrap:wrap;gap:.5rem;}
.hero .dato{border:1px solid rgba(143,211,220,.35);color:#D6EBEE;border-radius:999px;
  padding:.28rem .85rem;font-size:.84rem;}
.hero img{width:100%;display:block;}
@media (max-width:820px){.hero{grid-template-columns:1fr;padding:1.8rem 1.5rem;}
  .hero img{display:none;} .hero h1{font-size:2rem;}}

.eyebrow{display:flex;align-items:center;gap:.6rem;color:var(--dorado);font-size:.75rem;
  font-weight:700;letter-spacing:.16em;text-transform:uppercase;margin-bottom:1rem;}
.eyebrow::before{content:"";width:26px;height:2px;background:var(--dorado);}

.funciones{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:0 0 1.3rem 0;}
.func{background:#fff;border:1px solid var(--linea);border-radius:16px;padding:1rem 1.2rem;
  display:flex;gap:.95rem;align-items:center;
  box-shadow:0 1px 2px rgba(14,47,58,.04),0 6px 18px rgba(14,47,58,.05);}
.func img{width:46px;height:46px;flex:none;}
.func b{display:block;font-family:var(--serif);font-size:1.05rem;font-weight:600;color:var(--tinta);}
.func span{font-size:.86rem;color:var(--tenue);line-height:1.4;display:block;margin-top:.1rem;}
@media (max-width:820px){.funciones{grid-template-columns:1fr;}}

.aviso{display:flex;gap:.8rem;align-items:flex-start;background:#FFF8E8;
  border:1px solid #F0DDAE;border-left:5px solid var(--dorado);border-radius:10px;
  padding:.85rem 1.1rem;color:#5A4511;font-size:.95rem;line-height:1.5;margin-bottom:1rem;}

/* ---------- barra lateral ---------- */
[data-testid="stSidebar"]{background:var(--tinta);}
[data-testid="stSidebar"] *{color:#C9DEE2;}
[data-testid="stSidebar"] .marca{display:flex;align-items:center;gap:.7rem;margin:.2rem 0 1.1rem 0;}
[data-testid="stSidebar"] .marca b{font-family:var(--serif);font-size:1.25rem;color:#F2F8F9;font-weight:600;}
[data-testid="stSidebar"] .marca img{width:38px;height:38px;}
[data-testid="stSidebar"] h4{font-family:var(--serif);font-weight:500;color:#F2F8F9;
  font-size:1rem;margin:1.2rem 0 .5rem 0;}
.estado{display:flex;align-items:center;gap:.6rem;padding:.55rem .7rem;border-radius:8px;
  background:rgba(143,211,220,.08);margin-bottom:.4rem;font-size:.9rem;}
.estado i{width:8px;height:8px;border-radius:50%;background:#58C4A0;flex:none;}
.estado i.no{background:var(--dorado);}
.estado em{margin-left:auto;font-style:normal;opacity:.75;font-size:.8rem;}
[data-testid="stSidebar"] small{opacity:.7;line-height:1.5;display:block;}

/* ---------- pestañas ---------- */
.stTabs [data-baseweb="tab-list"]{gap:.3rem;border-bottom:1px solid var(--linea);}
.stTabs [data-baseweb="tab"]{height:auto;padding:.75rem 1.1rem;background:transparent;}
.stTabs [data-baseweb="tab"] p{font-weight:600;color:var(--tenue);font-size:1rem;}
.stTabs [aria-selected="true"] p{color:var(--tinta);}
.stTabs [data-baseweb="tab-highlight"]{background:var(--agua);height:3px;}
.stTabs [data-baseweb="tab-border"]{display:none;}

/* ---------- títulos de sección ---------- */
.sec{margin:1.4rem 0 .9rem 0;}
.sec h2{font-family:var(--serif);font-weight:600;font-size:1.55rem;color:var(--tinta);
  margin:0;padding:0;letter-spacing:-.005em;}
.sec p{color:var(--tenue);margin:.25rem 0 0 0;font-size:.97rem;max-width:70ch;}

/* ---------- formulario ---------- */
[data-testid="stForm"]{border:none;padding:0;background:transparent;}
[class*="st-key-card_"]{background:#fff;border:1px solid var(--linea);border-radius:16px;
  padding:1.2rem 1.3rem 1.3rem 1.3rem;
  box-shadow:0 1px 2px rgba(14,47,58,.04),0 6px 20px rgba(14,47,58,.05);}
.grupo{font-family:var(--serif);font-size:1.12rem;font-weight:600;color:var(--tinta);
  display:flex;align-items:center;gap:.6rem;margin:.1rem 0 .4rem 0;}
.grupo::before{content:"";width:4px;height:1.15rem;border-radius:2px;background:var(--agua);}
.stApp label p{color:var(--tinta);font-weight:500;font-size:.92rem;}
[data-baseweb="select"]>div,[data-baseweb="input"]>div,[data-baseweb="base-input"]{
  background:#fff!important;border-color:var(--linea)!important;border-radius:9px!important;}
[data-baseweb="select"] *,[data-baseweb="input"] input{color:var(--tinta)!important;}
[data-baseweb="select"]>div:hover,[data-baseweb="input"]>div:hover{border-color:var(--agua-2)!important;}

/* ---------- botones ---------- */
.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"]>button{
  border-radius:10px;font-weight:600;padding:.6rem 1.4rem;border:1px solid var(--agua);
  background:#fff;color:var(--agua);transition:background .15s,color .15s;}
.stButton>button:hover,.stDownloadButton>button:hover,
[data-testid="stFormSubmitButton"]>button:hover{background:var(--bruma);border-color:var(--agua);color:var(--tinta);}
button[kind="primary"],button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primaryFormSubmit"]{background:var(--agua)!important;
  color:#fff!important;border-color:var(--agua)!important;}
button[kind="primary"]:hover,button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="stBaseButton-primaryFormSubmit"]:hover{background:var(--tinta)!important;
  border-color:var(--tinta)!important;color:#fff!important;}

/* ---------- botón de enviar, menús, pasos, expanders ---------- */
[data-testid="stFormSubmitButton"]>button{width:100%;padding:.9rem 1.4rem;font-size:1.02rem;margin-top:.4rem;}
[data-baseweb="popover"] ul,[data-baseweb="popover"] [role="listbox"]{background:#fff!important;}
[data-baseweb="popover"] li,[data-baseweb="popover"] li *{color:var(--tinta)!important;}
[data-baseweb="popover"] li:hover,[data-baseweb="popover"] li[aria-selected="true"]{
  background:var(--bruma)!important;}
[data-testid="stNumberInputStepUp"],[data-testid="stNumberInputStepDown"]{
  background:var(--bruma)!important;color:var(--tinta)!important;}
[data-testid="stExpander"]{background:#fff;border:1px solid var(--linea);border-radius:14px;overflow:hidden;}
[data-testid="stExpander"] details{border:none!important;}
[data-testid="stExpander"] summary p{font-weight:600;color:var(--tinta);}
[data-testid="stSidebar"] .guia{margin:0;padding-left:1.1rem;font-size:.86rem;line-height:1.65;opacity:.9;}
.sp-chk{height:1.85rem;}
[data-testid="stElementContainer"]:has(iframe[height="0"]){height:0;overflow:hidden;margin:0;}

/* ---------- tarjetas de resultado ---------- */
.res{background:#fff;border:1px solid var(--linea);border-top:4px solid var(--agua);
  border-radius:16px;padding:1.2rem 1.3rem 1.3rem 1.3rem;height:100%;
  box-shadow:0 1px 2px rgba(14,47,58,.04),0 8px 24px rgba(14,47,58,.07);}
.res.alerta{border-top-color:var(--baya);} .res.ok{border-top-color:var(--pino);}
.res-t{font-family:var(--serif);font-size:1.2rem;font-weight:600;color:var(--tinta);}
.medidor{position:relative;width:100%;max-width:270px;margin:.6rem auto .2rem auto;}
.medidor img{display:block;width:100%;}
.medidor-v{position:absolute;left:0;right:0;bottom:1px;text-align:center;
  font-family:var(--serif);font-size:2.35rem;font-weight:600;line-height:1;color:var(--tinta);}
.medidor-v small{font-size:1.1rem;font-weight:500;margin-left:.1rem;color:var(--tenue);}
.leyenda{display:flex;justify-content:center;align-items:center;gap:.45rem;font-size:.82rem;
  color:var(--tenue);margin-bottom:.8rem;}
.leyenda i{width:3px;height:.95rem;background:var(--tinta);border-radius:2px;display:inline-block;}
.pildora{display:inline-block;padding:.22rem .75rem;border-radius:999px;font-size:.82rem;font-weight:600;}
.pildora.alerta{background:#F8E3E7;color:#8F2D40;} .pildora.ok{background:#DDF1E8;color:#1F6A4F;}
.pildora.info{background:#DDEFF3;color:#175A69;}
.res-s{font-size:.9rem;color:var(--tenue);margin:.75rem 0 0 0;line-height:1.45;}
.res-u{font-size:.9rem;color:var(--tinta);margin:.45rem 0 0 0;line-height:1.45;font-weight:500;}
.semanas{font-family:var(--serif);font-size:2.35rem;font-weight:600;color:var(--tinta);
  margin:.7rem 0 1.1rem 0;line-height:1;}
.semanas small{font-size:1.05rem;font-weight:500;color:var(--tenue);margin-left:.25rem;}
.regla{position:relative;display:flex;height:14px;border-radius:99px;overflow:hidden;
  background:var(--bruma);margin:0 0 .35rem 0;}
.regla .z1{background:#F1D9A0;} .regla .z2{background:#A9DCC8;} .regla .z3{background:#CFD9DC;}
.regla-wrap{position:relative;}
.banda{position:absolute;top:0;height:14px;background:rgba(14,47,58,.22);}
.aguja{position:absolute;top:-5px;width:4px;height:24px;margin-left:-2px;border-radius:2px;
  background:var(--tinta);box-shadow:0 0 0 2px #fff;}
.regla-et{display:flex;font-size:.7rem;color:var(--tenue);margin-bottom:.9rem;}
.regla-et span{display:block;white-space:nowrap;overflow:visible;}

.vacio{background:#fff;border:1.5px dashed var(--linea);border-radius:16px;padding:1.8rem;
  display:flex;align-items:center;gap:1.6rem;margin-top:1rem;}
.vacio img{width:120px;flex:none;}
.vacio h4{font-family:var(--serif);font-size:1.2rem;margin:0 0 .3rem 0;color:var(--tinta);}
.vacio p{margin:0;color:var(--tenue);max-width:55ch;line-height:1.5;}

/* ---------- pasos y paneles ---------- */
.pasos{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:.4rem 0 1.2rem 0;}
.paso{background:#fff;border:1px solid var(--linea);border-radius:14px;padding:1rem 1.1rem;
  display:flex;gap:.8rem;align-items:flex-start;}
.paso b{flex:none;width:28px;height:28px;border-radius:50%;background:var(--agua);color:#fff;
  display:flex;align-items:center;justify-content:center;font-size:.9rem;}
.paso span{font-size:.93rem;line-height:1.45;color:var(--tinta);}
@media (max-width:820px){.pasos{grid-template-columns:1fr;}}
[data-testid="stFileUploaderDropzone"]{background:var(--bruma);border:1.5px dashed var(--agua-2);
  border-radius:14px;}
[data-testid="stMetric"]{background:#fff;border:1px solid var(--linea);border-radius:14px;
  padding:.9rem 1.1rem;box-shadow:0 1px 2px rgba(14,47,58,.04);}
[data-testid="stMetricValue"]{font-family:var(--serif);color:var(--tinta);}
[data-testid="stMetricLabel"] p{color:var(--tenue);}

.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:.6rem 0 1.4rem 0;}
.kpi{background:var(--tinta);border-radius:16px;padding:1.3rem 1.4rem;}
.kpi b{display:block;font-family:var(--serif);font-size:2.3rem;font-weight:600;color:#F2F8F9;line-height:1.1;}
.kpi span{color:#B9D3D8;font-size:.92rem;}
@media (max-width:820px){.kpis{grid-template-columns:1fr;}}
.tabla{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--linea);
  border-radius:14px;overflow:hidden;font-size:.93rem;}
.tabla th{background:var(--bruma);color:var(--tinta);text-align:left;padding:.7rem .9rem;font-weight:600;}
.tabla td{padding:.7rem .9rem;border-top:1px solid var(--linea);color:var(--tinta);}
.paneles{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin:1.2rem 0;}
.panel{background:#fff;border:1px solid var(--linea);border-radius:16px;padding:1.2rem 1.4rem;}
.panel h4{font-family:var(--serif);font-size:1.15rem;margin:0 0 .6rem 0;color:var(--tinta);}
.panel ul{margin:0;padding-left:1.1rem;} .panel li{margin-bottom:.4rem;line-height:1.5;color:var(--tinta);}
@media (max-width:820px){.paneles{grid-template-columns:1fr;}}
.fichas{display:flex;flex-wrap:wrap;gap:.45rem;margin:.5rem 0 .4rem 0;}
.ficha{font-size:.82rem;padding:.25rem .7rem;border-radius:6px;border:1px solid var(--linea);
  background:#fff;color:var(--tinta);font-family:ui-monospace,Consolas,monospace;}
.ficha.num{background:var(--agua);border-color:var(--agua);color:#fff;}
.pie{margin-top:2.2rem;padding-top:1.1rem;border-top:1px solid var(--linea);color:var(--tenue);
  font-size:.85rem;line-height:1.6;}
</style>
"""


# =========================================================================== #
# ILUSTRACIONES (SVG incrustados; no dependen de archivos ni de internet)
# =========================================================================== #
def _img(svg: str, clase: str = "", alt: str = "") -> str:
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img class="{clase}" alt="{alt}" src="data:image/svg+xml;base64,{b64}">'


def svg_portada() -> str:
    puntos = "".join(
        f'<circle cx="{x}" cy="{y}" r="{r}" fill="{AGUA_CLARA}" fill-opacity="{o}"/>'
        for x, y, r, o in [(60, 40, 3, .5), (130, 215, 2.5, .4), (250, 30, 2, .5),
                           (430, 60, 3, .45), (440, 210, 2.5, .4), (30, 120, 2, .35),
                           (370, 235, 2, .4), (215, 235, 3, .3)])
    cruces = "".join(
        f'<path d="M{x-5} {y}H{x+5}M{x} {y-5}V{y+5}" stroke="{DORADO}" stroke-width="2" '
        f'stroke-linecap="round" stroke-opacity=".8"/>'
        for x, y in [(95, 80), (410, 140), (180, 25), (300, 225)])
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 460 260" fill="none">'
        f'<g stroke="#5FB3BF" stroke-width="1.5">'
        '<circle cx="330" cy="130" r="38" stroke-opacity=".5"/>'
        '<circle cx="330" cy="130" r="72" stroke-opacity=".35"/>'
        '<circle cx="330" cy="130" r="106" stroke-opacity=".22"/>'
        '<circle cx="330" cy="130" r="140" stroke-opacity=".12"/></g>'
        f'<circle cx="330" cy="130" r="24" fill="{AGUA}"/>'
        f'<circle cx="330" cy="130" r="11" fill="{DORADO}"/>'
        f'{puntos}{cruces}'
        f'<path d="M0 150H118L136 108L160 192L184 92L204 150H460" stroke="{AGUA_CLARA}" '
        'stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>')


def svg_marca() -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" fill="none">'
        f'<circle cx="24" cy="24" r="23" fill="{AGUA}"/>'
        '<circle cx="24" cy="24" r="15" stroke="#fff" stroke-opacity=".45" stroke-width="1.5"/>'
        '<path d="M6 26H17L21 17L27 33L31 24H42" stroke="#fff" stroke-width="2.4" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg>')


def svg_vacio() -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" fill="none">'
        '<circle cx="60" cy="60" r="58" fill="#EAF3F4"/>'
        '<rect x="30" y="22" width="60" height="78" rx="8" fill="#fff" stroke="#1F7A8C" stroke-width="2.5"/>'
        '<rect x="46" y="16" width="28" height="14" rx="5" fill="#1F7A8C"/>'
        '<path d="M40 52H80M40 64H68" stroke="#BFD6DB" stroke-width="3" stroke-linecap="round"/>'
        f'<path d="M38 84H52L57 74L64 92L69 82H82" stroke="{BAYA}" stroke-width="2.5" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg>')


def svg_medidor(p: float, u: float, color: str) -> str:
    largo = math.pi * 80
    ang = math.pi * (1 - u)
    x1, y1 = 100 + 66 * math.cos(ang), 100 - 66 * math.sin(ang)
    x2, y2 = 100 + 96 * math.cos(ang), 100 - 96 * math.sin(ang)
    arco = "M20 100A80 80 0 0 1 180 100"
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 112" fill="none">'
        f'<path d="{arco}" stroke="#DCE9EC" stroke-width="14" stroke-linecap="round"/>'
        f'<path d="{arco}" stroke="{color}" stroke-width="14" stroke-linecap="round" '
        f'stroke-dasharray="{max(p, 0.004) * largo:.2f} {largo + 30:.2f}"/>'
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{TINTA}" '
        'stroke-width="2.6" stroke-linecap="round"/></svg>')


# =========================================================================== #
# COMPONENTES HTML
# =========================================================================== #
def html_portada() -> str:
    return (
        '<div class="hero"><div>'
        '<div class="eyebrow">DANE · Estadísticas Vitales 2023</div>'
        '<h1>Riesgo perinatal en Colombia</h1>'
        '<p>Cesárea, bajo peso al nacer y tiempo de gestación estimados con las '
        'Estadísticas Vitales del DANE 2023, usando solo información disponible '
        'antes del parto.</p>'
        '<div class="datos"><span class="dato">508 776 nacimientos</span>'
        '<span class="dato">16 variables previas al parto</span>'
        '<span class="dato">Prueba usada una sola vez</span></div></div>'
        f'<div>{_img(svg_portada(), alt="")}</div></div>')


def html_titulo(titulo: str, detalle: str = "") -> str:
    sub = f"<p>{detalle}</p>" if detalle else ""
    return f'<div class="sec"><h2>{titulo}</h2>{sub}</div>'


def html_tarjeta_probabilidad(clave: str, p: float, u: float, uso: str) -> str:
    marcada = p >= u
    clase = "alerta" if marcada else "ok"
    color = BAYA if marcada else PINO
    estado = "Por encima del umbral" if marcada else "Por debajo del umbral"
    prev = PREVALENCIA[clave]
    return (
        f'<div class="res {clase}">'
        f'<div class="res-t">{OBJETIVOS[clave].nombre}</div>'
        f'<div class="medidor">{_img(svg_medidor(p, u, color), alt="")}'
        f'<div class="medidor-v">{p * 100:.1f}<small>%</small></div></div>'
        f'<div class="leyenda"><i></i>Umbral de decisión: {u * 100:.0f} %</div>'
        f'<span class="pildora {clase}">{estado}</span>'
        f'<p class="res-s">Equivale a {p / prev:.1f} veces la frecuencia general '
        f'({prev * 100:.1f} %).</p>'
        f'<p class="res-u">{uso}</p></div>')


def html_tarjeta_semanas(semanas: float) -> str:
    ini, fin = 24.0, 44.0
    pos = lambda w: min(max((w - ini) / (fin - ini), 0), 1) * 100  # noqa: E731
    izq, der = pos(semanas - RECM_SEMANAS), pos(semanas + RECM_SEMANAS)
    return (
        '<div class="res">'
        '<div class="res-t">Tiempo de gestación</div>'
        f'<div class="semanas">≈ {semanas:.1f}<small>semanas</small></div>'
        '<div class="regla-wrap"><div class="regla">'
        '<div class="z1" style="width:65%"></div><div class="z2" style="width:25%"></div>'
        '<div class="z3" style="width:10%"></div></div>'
        f'<div class="banda" style="left:{izq:.1f}%;width:{der - izq:.1f}%"></div>'
        f'<div class="aguja" style="left:{pos(semanas):.1f}%"></div></div>'
        '<div class="regla-et"><span style="width:65%">Pretérmino</span>'
        '<span style="width:25%">A término</span><span>42+</span></div>'
        '<span class="pildora info">Capacidad predictiva baja</span>'
        f'<p class="res-s">La franja sombreada marca el error típico del modelo '
        f'(±{RECM_SEMANAS:.1f} semanas). Pretérmino: menos de 37 semanas. '
        'El modelo explica solo 6.7 % de la variabilidad.</p>'
        '<p class="res-u">Sirve como referencia general, no para fechar un embarazo.</p></div>')


def html_vacio() -> str:
    return (
        f'<div class="vacio">{_img(svg_vacio(), alt="")}<div>'
        '<h4>Aún no hay resultados</h4>'
        '<p>Completa los datos de la gestante y presiona «Calcular predicción». '
        'Verás la probabilidad de cesárea y de bajo peso, y una estimación '
        'de las semanas de gestación.</p></div></div>')


def svg_icono(tipo: str) -> str:
    formas = {
        "cesarea": ("#DDEEF2",
                    f'<path d="M24 14V34M14 24H34" stroke="{AGUA}" stroke-width="4" '
                    'stroke-linecap="round"/>'),
        "bajo_peso": ("#F9E8EC",
                      f'<path d="M12 31A12 12 0 0 1 36 31" stroke="{BAYA}" stroke-width="3.5" '
                      'stroke-linecap="round"/>'
                      f'<path d="M24 31L29 22" stroke="{TINTA}" stroke-width="3" stroke-linecap="round"/>'
                      f'<circle cx="24" cy="31" r="2.6" fill="{TINTA}"/>'),
        "semanas": ("#FBF1DC",
                    f'<rect x="13" y="15" width="22" height="20" rx="4" stroke="{DORADO}" stroke-width="2.6"/>'
                    f'<path d="M13 22H35M19 12V17M29 12V17" stroke="{DORADO}" stroke-width="2.6" '
                    'stroke-linecap="round"/>'
                    + "".join(f'<circle cx="{x}" cy="28" r="1.8" fill="{DORADO}"/>' for x in (19, 24, 29))),
    }
    fondo, forma = formas[tipo]
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" fill="none">'
            f'<circle cx="24" cy="24" r="24" fill="{fondo}"/>{forma}</svg>')


def html_funciones() -> str:
    items = [("cesarea", "Probabilidad de cesárea", "Apoya la planeación de camas y quirófanos."),
             ("bajo_peso", "Bajo peso al nacer", "Alerta para priorizar el control prenatal."),
             ("semanas", "Semanas de gestación", "Estimación general, de capacidad baja.")]
    return '<div class="funciones">' + "".join(
        f'<div class="func">{_img(svg_icono(k), alt="")}<div><b>{t}</b><span>{d}</span></div></div>'
        for k, t, d in items) + '</div>'


def html_pasos() -> str:
    pasos = ["Descarga la plantilla y llénala con tus registros, o usa el archivo "
             "original del DANE.",
             "Sube el archivo CSV o Excel. Las columnas se reconocen por su nombre.",
             "Revisa el resumen y descarga el CSV con las probabilidades y las clases."]
    return '<div class="pasos">' + "".join(
        f'<div class="paso"><b>{i}</b><span>{t}</span></div>'
        for i, t in enumerate(pasos, 1)) + '</div>'


def html_modelos() -> str:
    return (
        '<div class="kpis">'
        '<div class="kpi"><b>0.723</b><span>AUC de cesárea en prueba</span></div>'
        '<div class="kpi"><b>29.3 %</b><span>de los casos de bajo peso detectados</span></div>'
        '<div class="kpi"><b>6.7 %</b><span>de la variabilidad de las semanas explicada</span></div></div>'
        '<table class="tabla"><thead><tr><th>Objetivo</th><th>Modelo</th><th>Umbral</th>'
        '<th>AUC</th><th>Casos detectados</th><th>Precisión</th><th>Reducción del costo</th></tr></thead>'
        '<tbody><tr><td>Cesárea</td><td>Random Forest</td><td>0.50</td><td>0.723</td>'
        '<td>62.3 %</td><td>67.6 %</td><td>31.8 %</td></tr>'
        '<tr><td>Bajo peso</td><td>Regresión logística</td><td>0.16</td><td>0.684</td>'
        '<td>29.3 %</td><td>27.6 %</td><td>13.9 %</td></tr>'
        '<tr><td>Tiempo de gestación</td><td>Regresión lineal múltiple</td><td>—</td>'
        '<td colspan="4">RECM 3.09 semanas · R² 0.067</td></tr></tbody></table>'
        '<p style="font-size:.85rem;color:#5B7480;margin-top:.6rem">Fuente: artículo, '
        'Sección 3.6 y Tabla 34. 101 756 nacimientos que los modelos nunca habían visto; '
        'los costos supuestos son 1 y 1 en cesárea y 1 y 5 en bajo peso.</p>'
        '<div class="paneles"><div class="panel"><h4>Uso previsto</h4><ul>'
        '<li><b>Bajo peso:</b> alerta para decidir a quién se le hace un control prenatal más cercano.</li>'
        '<li><b>Cesárea:</b> planeación de camas y quirófanos.</li>'
        '<li><b>Tiempo de gestación:</b> referencia general; casi no se puede predecir con estas variables.</li>'
        '</ul><p style="font-size:.88rem;color:#5B7480;margin:.6rem 0 0 0">Para secretarías de '
        'salud y comités de calidad de la atención materna.</p></div>'
        '<div class="panel"><h4>Límites que conviene conocer</h4><ul>'
        '<li>Se probó solo con 2023; falta validar con otro año.</li>'
        '<li>El modelo de bajo peso discrimina peor en madres de 10 a 19 años y en no aseguradas.</li>'
        '<li>Las semanas son una aproximación: la base trae cinco intervalos.</li>'
        '<li>No hay información clínica del control prenatal.</li>'
        '<li>Describe asociaciones, no efectos causales.</li></ul></div></div>')


def html_variables() -> str:
    fichas = "".join(
        f'<span class="ficha{" num" if c in NUMERICAS else ""}">{c}</span>' for c in PREDICTORAS)
    return (f'<div class="fichas">{fichas}</div>'
            '<p style="font-size:.85rem;color:#5B7480">Relleno oscuro: numéricas. '
            'Contorno: categóricas (códigos del DANE).</p>')


def html_pie() -> str:
    return ('<div class="pie">Proyecto: predicción de cesárea, bajo peso y tiempo de gestación '
            '(DANE, EEVV 2023) · ML NRC 1484 · UTB 2026-II<br>Luisa Fernanda Acosta Romero, '
            'Isabel Sofia Buelvas de la Hoz, María José Espitatela Julio y Lineth Villalba '
            'Dominiquett.</div>')


# =========================================================================== #
# INTERFAZ
# =========================================================================== #
st.markdown(ESTILO, unsafe_allow_html=True)

# Evita que el navegador traduzca la página (p. ej. «Padre» -> «Capellán»).
components.html("""<script>
try {
  const d = window.parent.document;
  d.documentElement.lang = 'es';
  d.documentElement.setAttribute('translate', 'no');
  d.body.classList.add('notranslate');
  if (!d.querySelector('meta[name="google"]')) {
    const m = d.createElement('meta'); m.name = 'google'; m.content = 'notranslate';
    d.head.appendChild(m);
  }
} catch (e) {}
</script>""", height=0)


@st.cache_resource(show_spinner="Cargando modelos…")
def obtener_predictor() -> tuple[Predictor, dict[str, str]]:
    pred = Predictor(DIR_MODELOS)
    errores = pred.cargar_todos()
    return pred, errores


predictor, errores_carga = obtener_predictor()

if not predictor.disponibles:
    st.error(
        f"No se encontraron modelos en `{DIR_MODELOS}`. Ejecuta el bloque de "
        "guardado de la E4 en el .qmd para generar `modelo_cesarea.joblib`, "
        "`modelo_bajo_peso.joblib` y `modelo_tiempo_gestacion.joblib`.")
    st.stop()
for clave, detalle in errores_carga.items():
    st.error(f"No se pudo cargar el modelo de {OBJETIVOS[clave].nombre}: {detalle}")


def selector(col: str, titulo: str, defecto: str | None = None,
             ayuda: str | None = None, por_etiqueta: bool = False) -> str:
    opciones = predictor.opciones(col, ordenar_por_etiqueta=por_etiqueta)
    codigos = [c for c, _ in opciones] + [SIN_INFO]
    nombres = dict(opciones)
    nombres[SIN_INFO] = "Sin información"
    indice = codigos.index(defecto) if defecto in codigos else 0
    return st.selectbox(titulo, codigos, index=indice, format_func=nombres.get,
                        help=ayuda, key=f"campo_{col}")


def a_valor(codigo: str):
    return np.nan if codigo == SIN_INFO else codigo


def grupo(texto: str) -> None:
    st.markdown(f'<div class="grupo">{texto}</div>', unsafe_allow_html=True)


def tarjeta(nombre: str):
    """Tarjeta blanca con estilo propio (clase st-key-card_<nombre>)."""
    try:
        return st.container(key=f"card_{nombre}")
    except TypeError:  # Streamlit antiguo sin parámetro key
        return st.container(border=True)


def mostrar_resultado(res: pd.DataFrame, entrada: pd.DataFrame) -> None:
    st.markdown(html_titulo("Resultado", "Lectura del modelo para el caso ingresado."),
                unsafe_allow_html=True)
    columnas = st.columns(3)
    usos = {
        "cesarea": "Uso sugerido: planear camas y quirófanos.",
        "bajo_peso": "Uso sugerido: priorizar el control prenatal.",
    }
    pos = 0
    for clave in ("cesarea", "bajo_peso"):
        if f"prob_{clave}" in res:
            u = res.attrs["umbrales"][clave]
            columnas[pos].markdown(
                html_tarjeta_probabilidad(
                    clave, float(res[f"prob_{clave}"].iloc[0]), u, usos[clave]),
                unsafe_allow_html=True)
            pos += 1
    if "semanas_gestacion" in res:
        columnas[pos].markdown(
            html_tarjeta_semanas(float(res["semanas_gestacion"].iloc[0])),
            unsafe_allow_html=True)

    for aviso in res.attrs.get("avisos", []):
        st.warning(aviso)

    st.write("")
    with st.expander("Cómo leer este resultado"):
        st.markdown(
            "- La **probabilidad** ordena el riesgo: sirve para comparar casos entre sí, "
            "no para afirmar que algo ocurrirá.\n"
            "- El **umbral** (la marca negra del medidor) es el punto desde el cual el "
            "modelo marca el caso. Se eligió en la validación cruzada minimizando el costo "
            "de los errores.\n"
            "- El modelo de bajo peso detecta cerca de 3 de cada 10 casos y funciona peor "
            "en madres adolescentes y no aseguradas; úsalo como alerta, nunca como único criterio.\n"
            "- Las asociaciones que aprende el modelo no son causales.")
    with st.expander("Datos ingresados"):
        st.dataframe(entrada.T.rename(columns={0: "Valor"}).astype(str))

    salida = pd.concat([res.reset_index(drop=True), entrada.reset_index(drop=True)], axis=1)
    st.download_button("Descargar resultado (CSV)",
                       salida.to_csv(index=False).encode("utf-8-sig"),
                       "prediccion_individual.csv", "text/csv")


# ---- portada y aviso ------------------------------------------------------- #
st.markdown(html_portada(), unsafe_allow_html=True)
st.markdown(
    '<div class="aviso"><div><b>Herramienta de apoyo, no de diagnóstico.</b> Sirve para '
    'priorizar el control prenatal y planear recursos. No decide sobre una gestante en '
    'particular: esa decisión sigue siendo del personal de salud.</div></div>',
    unsafe_allow_html=True)

st.markdown(html_funciones(), unsafe_allow_html=True)

# ---- barra lateral --------------------------------------------------------- #
with st.sidebar:
    st.markdown(
        f'<div class="marca">{_img(svg_marca(), alt="")}<b>Riesgo perinatal</b></div>'
        '<small>Modelos de aprendizaje automático con datos vitales del DANE, 2023.</small>',
        unsafe_allow_html=True)
    st.markdown("#### Modelos cargados")
    filas = ""
    for m in predictor.estado():
        punto = "" if m["disponible"] else ' class="no"'
        umbral = f"<em>umbral {m['umbral']:.2f}</em>" if m["umbral"] is not None else ""
        filas += f'<div class="estado"><i{punto}></i>{m["nombre"]}{umbral}</div>'
    st.markdown(filas, unsafe_allow_html=True)
    st.markdown(
        "<small>Cada archivo guarda el preprocesamiento (imputación y codificación) "
        "junto con el modelo: recibe los datos tal como vienen del DANE.</small>",
        unsafe_allow_html=True)
    st.markdown("#### Guía rápida")
    st.markdown(
        '<ol class="guia"><li>Completa los datos de la gestante.</li>'
        '<li>Presiona «Calcular predicción».</li>'
        '<li>Lee el resultado y descárgalo si lo necesitas.</li></ol>',
        unsafe_allow_html=True)
    if not predictor.categorias_leidas_del_modelo():
        st.markdown(
            "<small>No se pudieron leer las categorías del pipeline; las opciones "
            "siguen el diccionario del DANE.</small>", unsafe_allow_html=True)

tab_individual, tab_lote, tab_modelos = st.tabs(
    ["Predicción individual", "Predicción por archivo", "Sobre los modelos"])

# ---- pestaña 1: individual ------------------------------------------------- #
with tab_individual:
    st.markdown(html_titulo(
        "Datos de la gestante",
        "Usa solo información conocida antes del parto. Si un dato no está disponible, "
        "elige «Sin información»."), unsafe_allow_html=True)

    with st.form("formulario"):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            with tarjeta("embarazo"):
                grupo("Embarazo y recién nacido")
                sexo = selector("SEXO", "Sexo del recién nacido", "1")
                mul = selector("MUL_PARTO", "Tipo de embarazo", "1")
                n_consul = st.number_input(
                    "Consultas prenatales", 0, 40, 7, 1,
                    help="Total al final del control prenatal. Si el embarazo aún es "
                         "temprano, el modelo tenderá a subestimar el tiempo de gestación.")
                a, b = st.columns(2)
                n_emb = a.number_input("Embarazos", 1, 25, 2, 1,
                                       help="Cuenta también el embarazo actual.")
                n_hijos = b.number_input("Hijos nacidos vivos", 1, 25, 2, 1,
                                         help="Cuenta también a este recién nacido.")
            with tarjeta("residencia"):
                grupo("Residencia y seguridad social")
                dep = selector("CODPTORE", "Departamento de residencia", "13",
                               por_etiqueta=True)
                area = selector("AREA_RES", "Área de residencia", "1")
                seg = selector("SEG_SOCIAL", "Seguridad social", "2")
        with c2:
            with tarjeta("madre"):
                grupo("Madre")
                edad_m = selector("EDAD_MADRE", "Edad de la madre", "4")
                civil = selector("EST_CIVM", "Estado conyugal", "1")
                niv_m = selector("NIV_EDUM", "Nivel educativo", "4")
                cur_m = selector("ULTCURMAD", "Último año o grado aprobado",
                                 ayuda="Año o grado dentro del nivel educativo elegido (por ejemplo, grado 5 de primaria). 0 = ninguno.")
                etnia = selector("IDPERTET", "Pertenencia étnica", "6",
                                 ayuda="Variable sensible: se incluye porque el modelo la "
                                       "usa, no para tratar distinto a ningún grupo.")
            with tarjeta("padre"):
                grupo("Padre")
                a, b = st.columns(2)
                edad_p = a.number_input("Edad del padre (años)", 12, 90, 30, 1)
                with b:
                    st.markdown('<div class="sp-chk"></div>', unsafe_allow_html=True)
                    sin_edad_p = st.checkbox("Edad no disponible",
                                             help="Márcalo si no se conoce la edad del padre.")
                niv_p = selector("NIV_EDUP", "Nivel educativo", "4")
                cur_p = selector("ULTCURPAD", "Último año o grado aprobado",
                                 ayuda="Año o grado dentro del nivel educativo elegido (por ejemplo, grado 5 de primaria). 0 = ninguno.")
        enviar = st.form_submit_button("Calcular predicción", type="primary")

    if enviar:
        if n_hijos > n_emb and mul == "1":
            st.warning("Los hijos nacidos vivos superan los embarazos en un embarazo "
                       "simple. Revisa los datos antes de interpretar el resultado.")
        fila = {
            "NUMCONSUL": n_consul, "N_HIJOSV": n_hijos, "N_EMB": n_emb,
            "EDAD_PADRE": np.nan if sin_edad_p else edad_p,
            "SEXO": a_valor(sexo), "MUL_PARTO": a_valor(mul),
            "IDPERTET": a_valor(etnia), "EDAD_MADRE": a_valor(edad_m),
            "EST_CIVM": a_valor(civil), "NIV_EDUM": a_valor(niv_m),
            "ULTCURMAD": a_valor(cur_m), "CODPTORE": a_valor(dep),
            "AREA_RES": a_valor(area), "SEG_SOCIAL": a_valor(seg),
            "NIV_EDUP": a_valor(niv_p), "ULTCURPAD": a_valor(cur_p),
        }
        entrada = pd.DataFrame([fila], columns=PREDICTORAS)
        try:
            with st.spinner("Calculando…"):
                resultado = predictor.predecir(entrada)
            st.session_state["ultimo"] = (resultado, entrada)
        except (ValueError, RuntimeError, FileNotFoundError) as e:
            st.error(f"No se pudo calcular la predicción. {e}")

    if "ultimo" in st.session_state:
        mostrar_resultado(*st.session_state["ultimo"])
    else:
        st.markdown(html_vacio(), unsafe_allow_html=True)

# ---- pestaña 2: por archivo ------------------------------------------------ #
with tab_lote:
    st.markdown(html_titulo(
        "Predicción por archivo",
        "Calcula muchos registros a la vez. Puede traer las 39 columnas originales del "
        "DANE: solo se usan las 16 predictoras previas al parto."), unsafe_allow_html=True)
    st.markdown(html_pasos(), unsafe_allow_html=True)

    izq, der = st.columns([2, 1])
    with der:
        st.download_button("Descargar plantilla (CSV)",
                           plantilla_registros().to_csv(index=False).encode("utf-8-sig"),
                           "plantilla_registros.csv", "text/csv")
        limpiar = st.checkbox("Convertir 9 / 99 / 999 en «sin información»", value=True,
                              help="El DANE registra los datos desconocidos con estos "
                                   "códigos. Desactívalo solo si ya los limpiaste.")
    with izq:
        archivo = st.file_uploader("Archivo de registros", type=["csv", "xlsx", "xls"])

    if archivo is not None:
        try:
            datos = leer_registros(archivo)
            st.caption(f"{len(datos):,} registros leídos".replace(",", " "))
            with st.spinner("Calculando predicciones…"):
                res = predictor.predecir(datos, limpiar_codigos=limpiar)
        except (ValueError, RuntimeError, FileNotFoundError) as e:
            st.error(str(e))
        else:
            for aviso in res.attrs.get("avisos", []):
                st.warning(aviso)
            st.markdown(html_titulo("Resumen"), unsafe_allow_html=True)
            metricas = st.columns(4)
            metricas[0].metric("Registros", f"{len(res):,}".replace(",", " "))
            pos = 1
            for clave in ("cesarea", "bajo_peso"):
                if f"clase_{clave}" in res:
                    pct = (res[f"clase_{clave}"] == OBJETIVOS[clave].clase_positiva).mean()
                    metricas[pos].metric(f"{OBJETIVOS[clave].nombre}: marcados", f"{pct:.1%}")
                    pos += 1
            if "semanas_gestacion" in res:
                metricas[pos].metric("Semanas estimadas (media)",
                                     f"{res['semanas_gestacion'].mean():.1f}")

            probs = [c for c in res.columns if c.startswith("prob_")]
            if probs:
                elegido = st.selectbox(
                    "Distribución de probabilidades", probs,
                    format_func=lambda c: OBJETIVOS[c.removeprefix("prob_")].nombre)
                bordes = np.linspace(0, 1, 21)
                conteo, _ = np.histogram(res[elegido], bins=bordes)
                st.bar_chart(pd.DataFrame({"registros": conteo},
                                          index=[f"{b:.2f}" for b in bordes[:-1]]),
                             color=AGUA)

            tabla = res.copy()
            tabla.insert(0, "fila", np.arange(1, len(tabla) + 1))
            st.dataframe(tabla.head(500), hide_index=True)
            if len(tabla) > 500:
                st.caption("Se muestran las primeras 500 filas; el CSV incluye todas.")
            st.download_button("Descargar predicciones (CSV)",
                               tabla.to_csv(index=False).encode("utf-8-sig"),
                               "predicciones.csv", "text/csv", type="primary")

# ---- pestaña 3: modelos ---------------------------------------------------- #
with tab_modelos:
    st.markdown(html_titulo(
        "Modelos finales y desempeño en prueba",
        "Resultados de la evaluación única con nacimientos que los modelos no habían visto."),
        unsafe_allow_html=True)
    st.markdown(html_modelos(), unsafe_allow_html=True)
    st.markdown(html_titulo("Variables de entrada"), unsafe_allow_html=True)
    st.markdown(html_variables(), unsafe_allow_html=True)

st.markdown(html_pie(), unsafe_allow_html=True)
