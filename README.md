# Failure Prediction API & Streamlit Dashboard

This repository provides a production-ready **Failure Prediction API**, built with **FastAPI**, and an accompanying **Streamlit** dashboard for exploring and visualizing predictions. The machine learning model is trained and managed via Python scripts and pipelines in the `training/` folder, with artifacts persisted in `training/artifacts/`.

---

## 🗂️ Repository Structure

```
failure-prediction-api/
├── api/                      # FastAPI application
│   ├── controllers/          # Request handlers (health, predict, train)
│   ├── domain/               # Core entities and business models
│   ├── repositories/         # Data access layer (model loading/saving)
│   ├── routes/               # FastAPI route definitions
│   ├── services/             # Business logic (inference & training)
│   ├── utils/                # Utility functions (feature engineering, sequence helpers)
│   ├── dependencies.py       # Global dependency declarations
│   ├── main.py               # FastAPI app instantiation
│   └── Dockerfile            # Docker image spec for API
├── streamlit_app/            # Streamlit dashboard to consume the API
│   ├── app.py
│   └── Dockerfile
├── training/                 # Model training pipelines & scripts
│   ├── pipelines/            # Data & training pipeline implementations
│   ├── train_model.py        # Orchestrator for end-to-end training
│   └── artifacts/            # Generated artifacts (model, scaler, metadata)
├── data.xlsx                 # Sample dataset (for local experimentation)
├── tests/                    # Unit and integration tests
│   ├── unit/
│   └── integration/
├── docker-compose.yml        # Local development with Docker Compose
├── pyproject.toml            # Project metadata & dependencies (Poetry)
└── README.md                 # This documentation
```

---

## 🚀 Features

* **FastAPI** endpoints for:

  * **Health check** (`GET /health`)
  * **Single prediction** (`POST /predict/single`)
  * **Model training** (`POST /train`)
* **Streamlit** front-end to interactively send data and visualize results.
* Modular design with clear separation:

  * Controllers, routes, services, repositories, utilities.
* **Poetry**-based dependency management.
* Dockerized for consistent local development and easy deployment.
* Test suite covering controllers, services, repositories, and utils.

---

## 🛠️ Prerequisites

* [Docker & Docker Compose](https://docs.docker.com/compose/) (v2)
* [Poetry](https://python-poetry.org/) (optional, for local dev without Docker)
* Python 3.11+ (if running locally)

---

## ⚙️ Environment Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/gslmota/equipment-failure-prediction.git
   cd equipment-failure-prediction
   ```

2. **(Optional) Create `.env` file**
   Copy `.env.example` to `.env` (if provided) and adjust variables. At minimum, define:

   ```dotenv
   # .env
    PYTHONDONTWRITEBYTECODE=1
    TZ=America/Sao_Paulo
    TRAINING_DATA_PATH=/app/data.xlsx
    MODEL_ARTIFACTS_PATH=/app/training/artifacts
    LOG_LEVEL=INFO
    API_PORT=8000
   ```

3. **Install dependencies with Poetry** (for local development)

   ```bash
   poetry install
   ```

---

## 🐳 Running with Docker Compose

This is the recommended way for local development and testing:

```bash
docker compose up --build
```

* **API** will be available at `http://localhost:8000`
* **Swagger docs** at `http://localhost:8000/docs`
* **Streamlit** dashboard at `http://localhost:8501`

To tear down:

```bash
docker compose down
```

---

## 💻 Running Locally (Without Docker)

1. Activate your Poetry shell:

   ```bash
   poetry shell
   ```

2. **Start the API**

   ```bash
   uvicorn api.main:app --reload --host 0.0.0.0 --port $API_PORT
   ```

3. **Start the Streamlit app** (in a separate terminal):

   ```bash
   export API_URL=\"http://localhost:$API_PORT/predict/single\"
   streamlit run streamlit_app/app.py
   ```

---

## 🐞 Debugging

* **API logs**: By default, Uvicorn prints logs. Use `LOG_LEVEL=debug` in your `.env` or pass `--log-level debug`:

  ```bash
  uvicorn api.main:app --reload --log-level debug
  ```

* **Breakpoints**: Insert `import pdb; pdb.set_trace()` in code, or use your IDE’s debugger attached to the running process.

* **Rebuild Docker images** if you change dependencies or the Dockerfiles:

  ```bash
  docker compose build --no-cache api
  ```

---

## 🧪 Testing

All tests are based on **pytest**.

Run the full suite:

```bash
pytest
```

* **Unit tests** live in `tests/unit/`
* **Integration tests** live in `tests/integration/`

You can run a specific test module:

```bash
pytest tests/unit/services/test_inference_service.py
```

---

## 📝 Training a New Model

1. Prepare your data in `data.xlsx` (or point to a different source in your code).
2. Execute the training orchestration:

   ```bash
   python training/train_model.py --data-path data.xlsx
   ```
3. New artifacts (model, scaler, metadata) will be saved under `training/artifacts/`.
4. Restart the API to pick up the new model artifacts (if running as a service).

---

## 📈 Architecture & Design

1. **Controllers** handle HTTP requests and responses.
2. **Services** encapsulate business logic (feature engineering, training, inference).
3. **Repositories** manage persistence/loading of models and artifacts.
4. **Domain Entities** define data structures using Pydantic models.
5. **Utils** contain helper functions for sequence processing and feature pipelines.
6. **Streamlit App** consumes the API and visualizes predictions interactively.

---

## 📚 Further Improvements

* Add CI/CD pipeline (GitHub Actions, GitLab CI) for automated tests and Docker builds.
* Integrate model versioning (e.g., MLflow).
* Add authentication to the API.
* Implement more advanced feature-tracking and monitoring.

---

## 📄 License

This project is released under the MIT License. See `LICENSE` for details.
