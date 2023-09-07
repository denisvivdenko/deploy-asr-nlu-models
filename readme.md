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
    ```make init_env```
4. Activate the environment ```source venv/bin/activate```.
5. Run ```dvc pull```. You need to wait until the slurp dataset and models are loaded to your workdirectory.
6. (Optional) Run ```dvc repro```. You will run evaluation pipeline that uses SLURP dataset sample to evaluate final model performance.
7. Run ```docker-compose up --build``` to build a docker compose and start two endpoints.

Now there are two endpoints available:

    1. http://localhost:8000/docs  #ASR
    2. http://localhost:8001/docs  #NLU

FastAPI provides docs for each enpoint therefore you could try it out, upload your file and get the results.


# How this project works 

