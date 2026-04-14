# 🫀 ECG Anomaly Detection: Complete MLOps Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containers-blue.svg)](https://www.docker.com/)

## 📖 Introduction

This project demonstrates a production-grade **MLOps pipeline** for detecting anomalies in Electrocardiogram (ECG) data. It acts as an end-to-end Healthcare AI solution where a 1D Convolutional Neural Network (PyTorch) is deployed behind a highly scalable FastAPI server, containerized via Docker, and continuously monitored for latency and data drift using Prometheus and Grafana.

## 🛠️ Architecture and Technologies

1. **Deep Learning (PyTorch)**: A 1D-CNN designed to process 187-point time-series heartbeat signals.
2. **Automated Data Ingestion**: Uses `kagglehub` to seamlessly and automatically pull the MIT-BIH dataset from Kaggle directly into the training loop.
3. **Model Serving (FastAPI)**: REST API endpoint (`/predict`) executing model inference in a stateless, payload-strict environment.
4. **Monitoring / Observability (Prometheus & Grafana)**: Custom metrics tracking API latency, operational status, prediction distributions, and system throughput in real-time.
5. **Containerization**: Tied together with `docker-compose` for reproducible, clean, one-click local deployments.

## 🚀 Installation & Usage

### 1. Environment Setup

If you wish to test or train the model outside of Docker, install the dependencies securely using `uv`:

```bash
# Initialize uv virtual environment
uv venv

# Activate (Windows)
.\.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Train the Model

Train the 1D-CNN. The script will automatically fetch the required ~50MB data from Kaggle (ECG Heartbeat Categorization Dataset) on first run.

```bash
uv run src/train.py
```

### 3. Launch the MLOps Container Infrastructure

Deploy the FastAPI server side-by-side with Prometheus and Grafana instances using Docker Compose:

```bash
docker-compose up -d --build
```
- **API Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Grafana Dashboard**: [http://localhost:3000](http://localhost:3000) (Login: admin / admin)

### 4. Run the Load Simulator

Simulate high-throughput incoming hospital ECG data to view real-time Grafana metric shifts:

```bash
uv run scripts/simulate_traffic.py
```

## 📁 Repository Structure

```text
├── Dockerfile              # Minimal Docker image instruction setup
├── docker-compose.yml      # Orchestration for FastAPI, Prometheus, Grafana
├── prometheus.yml          # Scraper configuration for model tracking
├── requirements.txt        # Python dependency manifest
├── README.md               # You are here
├── data/                   # Automatically populated with dataset
├── models/                 # Serialized best model weights (.pth)
├── scripts/
│   └── simulate_traffic.py # Stress-testing and MLOps metrics generator
└── src/
    ├── api.py              # FastAPI endpoint w/ Prometheus instrumentation
    ├── data_loader.py      # PyTorch Dataset handling and Kaggle connection
    ├── model.py            # PyTorch 1D-CNN Network architecture
    └── retrain_job.py      # CI/CD training pipeline stub implementation
```

---

## 👨‍🔬 Author

**Sebastian Lijewski, PhD**
Portfolio Project for Machine Learning Engineering & MLOps
