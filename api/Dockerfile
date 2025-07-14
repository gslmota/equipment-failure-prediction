FROM python:3.11-slim
WORKDIR /app


RUN apt-get update && apt-get install -y libgomp1 \
  && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

COPY pyproject.toml README.md ./
COPY api/           ./api/
COPY training/      ./training/
COPY streamlit_app/ ./streamlit_app/

RUN poetry config virtualenvs.create false \
 && poetry install --without dev --no-interaction --no-ansi

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app/api"]

