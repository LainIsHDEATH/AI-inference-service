# Используем официальный PyTorch-образ с CUDA (здесь CUDA 12.1 и CUDNN 8, Python 3.10)
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# Копируем только requirements.txt и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код приложения (структуру app/)
COPY app/ ./app
COPY models/ ./models

# Открываем порт (7003)
EXPOSE 7003

# Запускаем сервер uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7003"]
