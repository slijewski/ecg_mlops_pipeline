FROM python:3.10-slim

WORKDIR /app

# Kopiujemy zaleznosci i instalujemy
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiujemy kod zrodlowy
COPY src/ src/

# Zmienne srodowiskowe pozwalajace na latwiejsze uzytkowanie modulu
ENV PYTHONPATH=/app

# Odkrywamy port dla FastAPI
EXPOSE 8000

# Komenda uruchamiajaca API przy uzyciu Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
