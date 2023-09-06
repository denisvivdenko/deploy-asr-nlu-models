FROM mosaicml/pytorch:1.13.1_cpu-python3.10-ubuntu20.04

WORKDIR /app

RUN pip install transformers==4.33.0 
# RUN pip install torch==2.0.1
RUN pip install onnxruntime==1.15.1

COPY models/* /app/models/
COPY endpoints/asr_model_endpoint/requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

COPY endpoints/asr_model_endpoint /app/
COPY src/utils/ /app/src/utils/
COPY params.yaml /app/params.yaml

EXPOSE 8000

CMD ["python3.10", "main.py"]