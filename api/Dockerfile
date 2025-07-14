# api/Dockerfile

FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy project metadata + code (all packages)
COPY pyproject.toml README.md ./
COPY api/ ./api/
COPY streamlit_app/ ./streamlit_app/
COPY training/ ./training/

# Install dependencies (will generate poetry.lock in‐image)
RUN poetry config virtualenvs.create false \
  && poetry install --without dev --no-interaction --no-ansi

EXPOSE 8000

# Start the FastAPI app via your console‐script
CMD ["poetry", "run", "start-api"]
