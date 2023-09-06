FROM python:3.10-slim-buster

WORKDIR /app

COPY models/* /app/models/
COPY endpoints/nlu_model_endpoint/requirements.txt /app/requirements.txt
COPY endpoints/nlu_model_endpoint /app/

RUN pip install -r requirements.txt

EXPOSE 8001

CMD ["python", "main.py"]