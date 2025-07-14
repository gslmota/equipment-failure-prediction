# streamlit_app/Dockerfile

FROM python:3.11-slim

WORKDIR /app

# System deps (needed by CatBoost—even if you don't call it here)
RUN apt-get update && apt-get install -y libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy project metadata + code (all packages)
COPY pyproject.toml README.md ./
COPY api/ ./api/
COPY streamlit_app/ ./streamlit_app/
COPY training/ ./training/

# Install dependencies
RUN poetry config virtualenvs.create false \
  && poetry install --without dev --no-interaction --no-ansi

EXPOSE 8501

# Run the Streamlit dashboard
CMD ["poetry", "run", "streamlit", "run", "streamlit_app/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
