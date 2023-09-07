# ASR+NLU Task

# High-level Functionality
1. Run ASR and NLU models on CPU
2. Deploy two endpoints
    - ASR model endpoint. Output JSON: 
    
    ```
    {
        "transcript": "..."
    }
    ```
    - NLU model endpoint. Output JSON: 
    ```
    {
        "transcript": "...",
        "intent": "..."
    }
    ```
3. RestAPI accepts files like Wav, mp3, flac
4. Automatic pipeline that evaluates models performance based on SLURP dataset.
5. Ability to easily change the models


# Stack

    Python == 3.10, DVC >= 3, Docker-compose, GCP (Google Drive), FastAPI, HuggingFace, ONNX

# How to run

Ubuntu:

1. Git clone the repo.
2. Put ```gcp-credentials.json``` file in your project root folder (I will send it).
3. Using Makefile. Run 
    ```make init_env``` (this project have to be installed using pip install -e .)
4. Activate the environment ```source venv/bin/activate```.
5. Run ```dvc fetch``` and after ```dvc pull```. You need to wait until the slurp dataset and models are loaded to your workdirectory. (dvc pull does different, but same thing, I guess I have caught a bug :), therefore running fetch might help)
6. (Optional) Run ```dvc repro```. You will run evaluation pipeline that uses SLURP dataset sample to evaluate final model performance.
7. Run ```docker-compose up --build``` to build a docker compose and start two endpoints.

Now there are two endpoints available:

    1. http://localhost:8000/docs  #ASR
    2. http://localhost:8001/docs  #NLU

FastAPI provides docs for each enpoint therefore you could try it out, upload your file and get the results.


# How this project works 

Pursuing such goals:
1. Be able easily evaluate any experiment and model.
2. Be to reproduce pipeline and results.
3. Keep track of metrics in github (Each run comments in pull requests and commits)

Therefore, for reproducibility I used DVC and Google Drive. 

If you want to change a model you need to make changes in params.yaml (if this is a minor change that doesn't change any logic, for example, another version of model in HuggingFace)

params.yaml contains all hyperparameters and model_names needed to run a pipeline.

Overall:
1. You make change in params.yaml
2. Push to repo
3. Everything runs in github actions and pushes the results and models to Google Drive.
4. You run ```git pull origin <you experiment branch>```. It will fetch new dvc.lock
5. You run ```dvc pull```, it fetches all dependencies to run your experiment.
6. You run docker-compose to test it using webapp.

You don't need to run it localy, you work won't be interrupted!


Project structure:
1. data/
2. endpoints/
    - model1/ - ASR
        - Dockerfile
    - model2/ - NLU model
        - Dockerfile
3. metrics/
    - metrics.json
4. models/
    - asr_model.onnx
    - nlu_model.onnx
5. notebooks/
6. src/
    - stages/
        pipeline stages
    - utils/
        utils for pipeline stages

