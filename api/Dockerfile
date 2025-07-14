FROM python:3.11-slim

WORKDIR /app

# Dependências de sistema
RUN apt-get update && apt-get install -y \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Instalar Poetry
RUN pip install poetry

# Copiar manifesto de dependências
COPY pyproject.toml ./

# Instalar dependências (vai gerar poetry.lock automaticamente)
RUN poetry config virtualenvs.create false \
  && poetry install --without dev --no-interaction --no-ansi

# Copiar código da API
COPY api/ ./api/

EXPOSE 8000

CMD ["poetry", "run", "start-api"]
