FROM python:3.12-slim

# uv installs the exact versions from uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /bin/uv

WORKDIR /app

# install dependencies first so Docker can cache this layer
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app.py .

ENV PATH="/app/.venv/bin:$PATH"
# the model is loaded from MLflow at startup; override this when running the container
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
