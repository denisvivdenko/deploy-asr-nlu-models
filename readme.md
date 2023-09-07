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



