FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# CMD is overridden per task by EventBridge:
#   core:   python core.py
#   broker: python broker.py
CMD ["python", "core.py"]
