FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["python", "-m", "phase0.rlm_annotator", "--replays", "data/replays", "--output", "data/annotations"]
