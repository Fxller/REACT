# 🧠 GenAI Reviews – Progetto SE4AI

[![Docker](https://img.shields.io/badge/docker-microservices-blue)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-ready-brightgreen)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-orange)](https://dagshub.com/Fxller/genai-reviews.mlflow)
[![DVC](https://img.shields.io/badge/DVC-enabled-purple)](https://dvc.org/)

Sistema a microservizi containerizzato per **generare** e **classificare** recensioni prodotto. Il tutto integrato con **FastAPI**, **MLflow** per l’MLOps e **DVC** per la gestione dei dati su **DagsHub**.

---

## ⚙️ Features

* ✨ **Generazione automatica** di recensioni da nome prodotto e rating
* 📊 **Classificazione automatica** di recensioni testuali
* 🚀 **API asincrone** con FastAPI
* 🔁 **Orchestrazione intelligente** via microservizi
* 🧪 **Monitoraggio esperimenti** ML con MLflow
* 🧺 **Gestione dati** tramite DVC (Dataset Amazon Reviews)

---

## 🧱 Architettura

```text
📦 genai-reviews/
🔼-- orchestrator/          # Riceve input e smista richieste
🔼-- reviews-generator/     # Genera la recensione testuale (porta 8001)
🔼-- reviews-classifier/    # Classifica la recensione (porta 8002)
🔼-- frontend/              # (Opzionale) interfaccia utente
🔼-- docker-compose.yml     # Orchestrazione container
🔼-- .env / .env.example    # Configurazioni ambiente
🔼-- data/, scripts/, utils/, .dvc/ ...
```

---

## ▶️ Setup del progetto

### 1. Clona la repository

```bash
git clone https://dagshub.com/Fxller/genai-reviews.git
cd genai-reviews
```

### 2. Crea il file `.env`

```bash
cp .env.example .env
```

Compila con le tue credenziali DagsHub:

```dotenv
MLFLOW_TRACKING_USERNAME=ilTuoUsername
MLFLOW_TRACKING_PASSWORD=ilTuoAccessToken
MLFLOW_TRACKING_URI=https://dagshub.com/Fxller/genai-reviews.mlflow
```

### 3. Avvia i container

```bash
docker-compose up --build
```

Per riavviare pulito:

```bash
docker-compose down
docker-compose up --build
```

---

## 🧪 Test delle API (via orchestrator - porta 8000)

### 🔀 Generazione recensione

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"prompt": "genera una recensione per un rossetto a 4 stelle"}'
```

### ⭐ Classificazione recensione

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"prompt": "valuta questa recensione: Il profumo è durato solo due ore"}'
```

---

## 📊 Monitoraggio esperimenti

Traccia ogni esecuzione su MLflow:
👉 [Visualizza su DagsHub](https://dagshub.com/Fxller/genai-reviews/experiments)

---

## 🧠 Modelli e tecnologie

* 🧬 `transformers`, `torch` – NLP pre-addestrato
* 🧪 `scikit-learn` – classificazione
* ⚡ `FastAPI` – microservizi REST asincroni
* 🔍 `MLflow` – logging e metriche
* 📦 `Docker`, `docker-compose` – containerizzazione
* 📂 `DVC` – gestione dataset

---

## 👥 Autori

* 🧑‍💻 Rosa Carotenuto
* 👩‍💻 Luigi Guida
* 🧑‍💻 Francesco Perilli
