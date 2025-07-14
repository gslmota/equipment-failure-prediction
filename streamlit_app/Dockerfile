FROM python:3.11-slim

WORKDIR /app

# Instalar Poetry
RUN pip install poetry

# Copiar manifesto de dependências
COPY pyproject.toml ./

# Instalar dependências (gera poetry.lock internamente)
RUN poetry config virtualenvs.create false \
  && poetry install --without dev --no-interaction --no-ansi

# Copiar app Streamlit
COPY streamlit_app/ ./streamlit_app/

EXPOSE 8501

CMD ["poetry", "run", "streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
