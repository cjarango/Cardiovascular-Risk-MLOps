# Sistema de Clasificación y Despliegue MLOps para Riesgo Cardiovascular

---

**Autores:**
* **Paula Andrea Gómez Vargas** (apaulag@uninorte.edu.co)
* **Juan Camilo Mendoza Arango** (cjarango@uninorte.edu.co)
* **Miguel Ángel Pérez Vargas** (vargasmiguel@uninorte.edu.co)

---

## Descripción general del proyecto

Este proyecto documenta y ejecuta el ciclo de vida completo de un modelo de Machine Learning (MLOps) diseñado para predecir el riesgo de enfermedad cardíaca. A diferencia de un ejercicio puramente estadístico, este repositorio abarca desde el diagnóstico clínico de los datos y la mitigación rigurosa de fuga de información (Data Leakage), hasta la serialización y puesta en producción del artefacto predictivo.

El flujo de trabajo integra la construcción de pipelines robustos, el empaquetado del modelo en un entorno aislado y su despliegue mediante una API REST, garantizando que el sistema pase de la fase de experimentación local a una arquitectura escalable y lista para integrarse en aplicaciones clínicas de tiempo real.

## Fuente de datos

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
* **Origen:** [Kaggle - Heart Failure Prediction Dataset](https://www.kaggle.com/fedesoriano/heart-failure-prediction)
* **Referencia:** fedesoriano. (September 2021). Heart Failure Prediction Dataset. Retrieved 30 de marzo del 2026 from https://www.kaggle.com/fedesoriano/heart-failure-prediction.

## Arquitectura y tecnologías utilizadas

Para garantizar la reproducibilidad y el despliegue continuo, el proyecto hace uso de las siguientes herramientas:

* **Scikit-Learn & Pandas:** Construcción del pipeline de preprocesamiento y entrenamiento del clasificador (Árboles de Decisión).
* **MyST (Markedly Structured Text):** Compilación y renderizado de la documentación estática a partir de libretas Jupyter (`_build`).
* **FastAPI & Pydantic:** Creación del servidor web y validación de esquemas de datos para inferencia.
* **Docker:** Contenerización del entorno de la API y sus dependencias.
* **Kubernetes (K8s):** Orquestación del contenedor local mediante despliegues y servicios balanceados.

## Estructura del repositorio

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
└── myst.yml
