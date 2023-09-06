FROM python:3.10-slim-buster

WORKDIR /app

COPY models/* /app/models/
COPY endpoints/asr_model_endpoint/requirements.txt /app/requirements.txt
COPY endpoints/asr_model_endpoint /app/

RUN pip install -r requirements.txt


