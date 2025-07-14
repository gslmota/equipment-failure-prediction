FROM python:3.9-slim

WORKDIR /app

# Instalar Poetry
RUN pip install poetry

# Copiar arquivos de dependências
COPY pyproject.toml poetry.lock ./

# Instalar dependências (sem dev)
RUN poetry config virtualenvs.create false && \
    poetry install --without dev --no-interaction --no-ansi

# Copiar aplicação Streamlit
COPY ./streamlit_app /app/streamlit_app

# Expor porta
EXPOSE 8501

# Comando para iniciar o Streamlit
CMD ["poetry", "run", "streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]