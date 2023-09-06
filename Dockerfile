FROM python:3.10-slim-buster

RUN pip install dvc==3.16.0

COPY . .

CMD ["dvc", "pull"]

# CMD ["python", "--version"]