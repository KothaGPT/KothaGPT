FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt /app/services/api/requirements.txt
RUN pip install --no-cache-dir -r /app/services/api/requirements.txt
COPY . /app/services/api
CMD ["uvicorn", "services.api.app:app", "--host", "0.0.0.0", "--port", "8000"]