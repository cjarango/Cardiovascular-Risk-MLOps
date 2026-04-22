# Sistema de Clasificación y Despliegue MLOps para Riesgo Cardiovascular

# Descripción general del proyecto

Este proyecto documenta y ejecuta el ciclo de vida completo de un modelo de Machine Learning (MLOps) diseñado para predecir el riesgo de enfermedad cardíaca. Toda la documentación técnica, los hallazgos del análisis exploratorio y los detalles de la arquitectura se encuentran disponibles y renderizados para su consulta en la página oficial del proyecto: https://cjarango.github.io/Cardiovascular-Risk-MLOps/.

A diferencia de un ejercicio puramente estadístico, este repositorio abarca desde el diagnóstico clínico de los datos y la mitigación rigurosa de fuga de información (*Data Leakage*), hasta la serialización y puesta en producción del artefacto predictivo. El flujo de trabajo integra la construcción de pipelines robustos, el empaquetado del modelo en un entorno aislado y su despliegue mediante una API REST, garantizando que el sistema pase de la fase de experimentación local a una arquitectura escalable y lista para integrarse en aplicaciones clínicas de tiempo real.

# Fuente de datos

El conjunto de datos utilizado es el **Heart Failure Prediction Dataset**, una colección consolidada y curada de 5 bases de datos cardiovasculares independientes (Cleveland, Hungarian, Switzerland, Long Beach VA y Stalog). Al unificar sus atributos comunes y eliminar duplicados, conforma el repositorio más grande disponible en su tipo para propósitos de investigación, contando con un total de **918 observaciones clínicas**.

Las enfermedades cardiovasculares (ECV) son la principal causa de muerte a nivel mundial. La detección temprana y el manejo preventivo mediante modelos predictivos son fundamentales para asistir el criterio médico en pacientes con alto riesgo.

**Diccionario de variables (Atributos clínicos):**
El conjunto consta de 11 características predictoras y 1 variable objetivo:

* **Age:** Edad del paciente [Años].
* **Sex:** Sexo del paciente [`M`: Masculino, `F`: Femenino].
* **ChestPainType:** Tipo de dolor de pecho [`TA`: Angina Típica, `ATA`: Angina Atípica, `NAP`: Dolor no anginoso, `ASY`: Asintomático].
* **RestingBP:** Presión arterial en reposo [mm Hg].
* **Cholesterol:** Colesterol sérico [mm/dl].
* **FastingBS:** Glucosa en sangre en ayunas [`1`: si > 120 mg/dl, `0`: en caso contrario].
* **RestingECG:** Resultados del electrocardiograma en reposo [`Normal`: Normal, `ST`: Anormalidad de la onda ST-T, `LVH`: Hipertrofia ventricular izquierda probable/definitiva según los criterios de Estes].
* **MaxHR:** Frecuencia cardíaca máxima alcanzada [Valor numérico entre 60 y 202].
* **ExerciseAngina:** Angina inducida por el ejercicio [`Y`: Sí, `N`: No].
* **Oldpeak:** Depresión del segmento ST inducida por el ejercicio en relación con el reposo [Valor numérico].
* **ST_Slope:** La pendiente del segmento ST durante el ejercicio máximo [`Up`: Ascendente, `Flat`: Plana, `Down`: Descendente].
* **HeartDisease (Target):** Clase de salida u objetivo clínico [`1`: Presencia de enfermedad cardíaca, `0`: Normal].

**Cita y Referencia:**

El conjunto de datos utilizado en este proyecto proviene del repositorio *Heart Failure Prediction Dataset* disponible en **Kaggle**. La referencia bibliográfica para este trabajo es:

* fedesoriano. (Septiembre, 2021). *Heart Failure Prediction Dataset*. Recuperado el 3 de abril de 2026 de https://www.kaggle.com/fedesoriano/heart-failure-prediction.

# Arquitectura y tecnologías utilizadas

Para garantizar la reproducibilidad y el despliegue continuo, el proyecto hace uso de las siguientes herramientas:

* **Scikit-Learn & Pandas:** Construcción del pipeline de preprocesamiento y entrenamiento del clasificador (Árboles de Decisión).
* **MyST (Markedly Structured Text):** Compilación y renderizado de la documentación estática a partir de libretas Jupyter (`_build`).
* **FastAPI & Pydantic:** Creación del servidor web y validación de esquemas de datos para inferencia.
* **Docker:** Contenerización del entorno de la API y sus dependencias.
* **Kubernetes (K8s):** Orquestación del contenedor local mediante despliegues y servicios balanceados.

# Estructura del repositorio

El proyecto está modularizado para separar claramente la experimentación, los datos, el código fuente y la infraestructura de despliegue:

```text
heart-disease-mlops/
├── .github/
├── _build/
├── app/
│   ├── api.py
│   └── model.joblib
├── data/
├── docker/
│   ├── Dockerfile
│   └── requirements.txt
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── monitoring/
│   ├── drift_report.html
│   └── generate_drift_report.py
├── notebooks/
│   ├── 01_EDA_y_Diagnóstico_de_Datos.ipynb
│   ├── 02_model_leakage_demo.ipynb
│   └── 03_model_pipeline_cv.ipynb
├── src/
│   ├── eda_toolkit.py
│   ├── model_evaluation_toolkit.py
│   └── visual_diagnostics_toolkit.py
├── .gitignore
├── custom.css
├── LICENSE
├── setup.cfg
└── myst.yml
```

# Monitoreo y mantenimiento del modelo

Como fase crítica del ciclo de vida del proyecto, se integró un protocolo de monitoreo proactivo mediante la librería `Evidently AI`. El análisis de deriva estadística se ejecutó comparando el conjunto de entrenamiento (*Reference*) frente al conjunto de prueba (*Current*), funcionando como una validación interna de estabilidad fundamental antes del despliegue. Este procedimiento asegura que el modelo no solo sea preciso en la fase de entrenamiento, sino que sus bases estadísticas permanezcan coherentes al enfrentarse a nuevos datos, estableciendo así una línea base robusta para la operación del sistema.

Los resultados del análisis demuestran una estabilidad global superior al $90\%$, lo que confirma la integridad del pipeline de datos. Se detectó una deriva estadísticamente significativa únicamente en la variable del sexo del paciente ($p < 0.05$ mediante Z-test); sin embargo, este comportamiento es plenamente consistente con la variabilidad esperada por el muestreo aleatorio durante la partición del dataset y no representa una degradación en la lógica predictiva.

En conclusión, la estabilidad observada en las variables clínicas de mayor peso, como el colesterol, la presión arterial y la frecuencia cardíaca máxima, es un indicador inequívoco de que el modelo generaliza adecuadamente. La ausencia de sesgos estructurales relevantes y la consistencia en las distribuciones de los predictores clave permiten validar técnicamente el artefacto. Por lo tanto, el sistema se considera maduro y se encuentra listo para su integración en entornos de producción bajo la arquitectura de `FastAPI` y `Docker`.

## Reporte de monitoreo

El sistema genera un dashboard interactivo que permite inspeccionar:

- Distribuciones de probabilidad  
- Distancias estadísticas  
- Variables con drift  

**Ubicación:**

```bash
monitoring/drift_report.html
```

**Ubicación:**

```bash
monitoring/generate_drift_report.py
```

# ¿Cómo ejecutar el proyecto?

Siga estos pasos para replicar el entorno de experimentación y despliegue:

## 1. Preparación del Entorno Local

Instale las dependencias optimizadas para evitar conflictos de versiones (NumPy < 2.0):

```bash
pip install -r docker/requirements.txt
```

## 2. Gestión de Monitoreo

Genere o actualice el reporte de deriva estadística basado en los últimos datos:

```bash
python monitoring/generate_drift_report.py
```

## 3. Contenerización con Docker

Construya y ejecute la API de inferencia en un entorno aislado:

```bash
docker build -t heart-disease-api -f docker/Dockerfile .
docker run -p 80:80 heart-disease-api
```

## 4. Orquestación con Kubernetes (K8s)

Despliegue el servicio de forma escalable en un clúster local (ej. Minikube):

```bash
kubectl apply -f k8s/
```