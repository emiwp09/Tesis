# Predicción de Parto Pretérmino mediante Aprendizaje Automático

## 📘 Repositorio de Trabajo de Tesis – Física (Ciencia de Datos)

Este repositorio constituye un espacio de trabajo académico que contiene los análisis computacionales, procesamiento de datos y modelos predictivos desarrollados como parte de una tesis de licenciatura en Física con orientación en Ciencia de Datos.

Su propósito es documentar de manera estructurada el flujo de trabajo y facilitar la reproducibilidad de los resultados presentados en la tesis.

---

## Contenido del Repositorio

### CSVtesis
Contiene los conjuntos de datos utilizados en el estudio:

- `alpha_diversity`  
- `metadata`  
- `CST (Community State Types)`  
- `valencia_taxons`  
- `phylotypes_nreads`  
- `taxonomy`  

Los tres primeros archivos pueden visualizarse directamente en GitHub. Los dos últimos fueron comprimidos/encriptados debido a su tamaño.

---

### Análisis Exploratorio de Datos

Los notebooks se organizan conforme a la estructura metodológica de la tesis.

#### 3.3.1 Exploración Clínica y Demográfica
Códigos 1–9 (`cod_#numero-Nombre.ipynb`), enfocados en la caracterización clínica y demográfica.

#### 3.3.2 Caracterización del Microbioma y Estructura Taxonómica
Códigos 10–17, enfocados en diversidad microbiana y estructura taxonómica. Incluye archivos derivados con las bacterias más frecuentes (Top 10 y Top 5).

#### 3.5 Modelos Implementados – Machine Learning
Notebooks de los modelos predictivos, nombrados como:

A#-Nombre_del_modelo.ipynb

Esta nomenclatura corresponde a la forma en que los modelos fueron referenciados en el apéndice de la tesis.

---

### Correcciones_Tesis
Versión corregida de los modelos, tras la revisión de la tesis. Los notebooks originales de la carpeta 3.5 se conservan sin cambios como referencia.

Principales correcciones:

- Los tres modelos usan la misma etiqueta (`was_preterm`) y se evalúan por paciente, con los mismos 25 pliegues (5 pliegues × 5 repeticiones).
- Ninguna paciente aparece a la vez en entrenamiento y prueba.
- El umbral de decisión se elige solo con datos de entrenamiento (índice de Youden), nunca con los de prueba.
- Se agregan controles e intervalos de confianza para sustentar las conclusiones sobre no linealidad y dinámica temporal.

Las instrucciones para correrlos están en `Correcciones_Tesis/LEEME.md`, y los resultados usados en la tesis en `Correcciones_Tesis/resultados/`.

## Formato de los Archivos

Todos los análisis se presentan en formato **Jupyter Notebook (.ipynb)**, lo que permite integrar código, resultados numéricos y visualizaciones en un solo documento.

---

## Entorno de Desarrollo

El proyecto fue desarrollado en **Python 3.11** utilizando principalmente:

- numpy  
- pandas  
- scikit-learn  
- tensorflow  
- matplotlib  
- seaborn  
- scipy  

Para reproducir el entorno:

pip install -r requirements.txt

El archivo `requirements.txt` incluido contiene las dependencias completas del entorno de trabajo.

---

## Objetivo

Este repositorio respalda técnicamente los resultados de la tesis y busca asegurar la reproducibilidad, trazabilidad y transparencia del análisis computacional realizado.

-Francisco Emiliano Jiménez Zamora
