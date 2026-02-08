FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install -U pip \
 && pip install -e .

EXPOSE 8000
CMD ["rag", "serve", "--config", "configs/rag.yaml"]
