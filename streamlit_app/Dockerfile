FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 \
  && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

COPY pyproject.toml README.md ./
COPY api/ ./api/
COPY streamlit_app/ ./streamlit_app/
COPY training/ ./training/
RUN poetry config virtualenvs.create false \
 && poetry install --without dev --no-interaction --no-ansi

EXPOSE 8501
CMD ["poetry", "run", "streamlit", "run", "streamlit_app/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
