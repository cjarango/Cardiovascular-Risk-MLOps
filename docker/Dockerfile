# 1. Usa una versión ligera de Python 3.10 como sistema operativo base
FROM python:3.10-slim

# 2. Define la carpeta /app como tu espacio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copia el archivo de requerimientos desde tu PC al contenedor
COPY docker/requirements.txt .

# 4. Instala las librerías sin guardar caché (hace la imagen más ligera)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia toda tu carpeta local 'app' a la carpeta '/app' del contenedor
COPY app/ ./app/

# 6. Comando para encender el servidor Uvicorn en el puerto 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]