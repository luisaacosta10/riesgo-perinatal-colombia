# Riesgo perinatal en Colombia

Aplicación interactiva que estima la probabilidad de cesárea, la probabilidad
de bajo peso al nacer y las semanas de gestación, usando las Estadísticas
Vitales del DANE 2023 y solo información disponible antes del parto.

**Aplicación en línea:** https://riesgo-perinatal-colombia-ticmcrywrgff6zhoiewgwz.streamlit.app/

## Modelos
- Cesárea: Random Forest (umbral 0.50)
- Bajo peso al nacer: regresión logística (umbral 0.16)
- Semanas de gestación: regresión lineal múltiple

## Cómo ejecutarla en tu computador
pip install -r requirements.txt
streamlit run app.py

## Aviso
Herramienta de apoyo, no de diagnóstico. La decisión sobre una gestante
sigue siendo del personal de salud.

## Autoras
Luisa Fernanda Acosta Romero, Isabel Sofia Buelvas de la Hoz,
María José Espitatela Julio y Lineth Villalba Dominiquett.
ML NRC 1484 · UTB 2026-II
