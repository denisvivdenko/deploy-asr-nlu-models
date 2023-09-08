init_env:
	python3.10 -m venv venv; \
	source venv/bin/activate; \
	pip install -e .; \

fetch_data:
	dvc fetch; \
	dvc pull; \

deploy_convert_asr_model_to_onnx:
	dvc stage add --force --name convert_asr_model_to_onnx \
	--outs models/asr_model.onnx \
	--params base,convert_asr_model \
	python src/stages/convert_asr_model_to_onnx.py --config="params.yaml"; \

deploy_convert_nlu_model_to_onnx:
	dvc stage add --force --name convert_nlu_model_to_onnx \
	--outs models/nlu_model.onnx \
	--params base,convert_nlu_model \
	python src/stages/convert_nlu_model_to_onnx.py --config="params.yaml"; \

deploy_preprocess_dataset:
	dvc stage add --force --name preprocess_dataset \
	--deps data/slurp_dataset \
	--outs data/preprocessed_dataset.csv \
	--params base,data_preprocessing \
	python src/stages/preprocess_dataset.py --config="params.yaml"; \

deploy_asr_inference:
	dvc stage add --force --name asr_inference \
	--deps models/asr_model.onnx \
	--outs data/asr_predictions.csv \
	--params base,data_preprocessing,convert_asr_model,asr_inference \
	python src/stages/run_asr_model.py --config="params.yaml"; \

deploy_nlu_inference:
	dvc stage add --force --name nlu_model \
	--deps models/nlu_model.onnx \
	--outs data/nlu_predictions.csv \
	--params base,data_preprocessing,convert_nlu_model,asr_inference \
	python src/stages/run_nlu_model.py --config="params.yaml"; \

deploy_evaluation:
	dvc stage add --force --name evaluation \
	--deps data/nlu_predictions.csv \
	--deps data/preprocessed_dataset.csv \
	--metrics metrics/metrics.json \
	--params base,data_preprocessing,convert_nlu_model,convert_asr_model,asr_inference,nlu_inference \
	python src/stages/evaluation.py --config="params.yaml"; \

deploy_all:
	make deploy_convert_asr_model_to_onnx; \
	make deploy_convert_nlu_model_to_onnx; \
	make deploy_preprocess_dataset; \
	make deploy_asr_inference; \
	make deploy_nlu_inference; \
	make deploy_evaluation; \