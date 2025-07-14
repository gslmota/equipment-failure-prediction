FROM python:3.9-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar Poetry
RUN pip install poetry

# Copiar arquivos de dependências
COPY pyproject.toml poetry.lock ./

# Instalar dependências (sem dev)
RUN poetry config virtualenvs.create false && \
    poetry install --without dev --no-interaction --no-ansi

# Copiar código da API
COPY ./api /app/api

# Expor porta
EXPOSE 8000

# Comando para iniciar a API
CMD ["poetry", "run", "start-api"]