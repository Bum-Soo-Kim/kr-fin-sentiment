FROM python:3.15-slim

WORKDIR /app
ENV POETRY_VIRTUALENVS_IN_PROJECT=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir poetry
COPY pyproject.toml poetry.lock poetry.toml ./
RUN poetry install --only main --no-root
COPY scoring.py finbert.py app.py ./

EXPOSE 8000
CMD ["poetry", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
