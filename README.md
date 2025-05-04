# Deep Learning Project: Summarization

## Project Overview
This project focuses on abstractive summarization of any kind of text (research paper abstracts, etc...) into concise titles using state-of-the-art deep learning models. The models fine-tuned include `mT5-small` and `BART-base`. The project also includes an API for generating summaries using the fine-tuned BART model.

## Features
- Fine-tuned abstractive summarization models.
- Baseline comparison using the Lead-3 method.
- API for summarization using FastAPI.
- Dockerized deployment for the API.

## Repository Structure
```

|── .gitignore
|── data/
│   └── top-10k-longest-abstracts.csv
|── api/
│   └── main
|── results/
│   ├── mt5-small-finetune-best/
│   └── bart-base-finetune-best/
├── final.ipynb
|── requirements.txt
|── README.md
```

**Explanation:**

*   **`.gitignore`**: Prevents committing unnecessary files/folders: __pycache__/, mlruns/).
*   **`data/`**: Stores the raw training dataset
*   **`final.ipynb`**: Contains the primary Jupyter/Colab notebooks for analysis, training, and evaluation.
*   **`results/`**: A place to store outputs like saved model checkpoints. It's often excluded from Git via `.gitignore` unless Git LFS is used for large files.
*   **`requirements.txt`**: Generated using `pip freeze > requirements.txt`. Ensures others can recreate the exact environment.
*   **`README.md`**: Description of the project for reproducibility:
    *   Project Title and Goal
    *   Dataset description and how to obtain it.
    *   Setup instructions (`pip install -r requirements.txt`).
    *   How to run the code (e.g., execute the notebook cells sequentially).
    *   Summary of results (baseline vs. models).
    *   Instructions on how to use the fine-tuned model (e.g., from Hugging Face Hub).
    *   Information about the repository structure.

2. Run the API Locally
Navigate to the api/ directory and run the FastAPI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

4. Access the API
Once the server is running, you can access the API at http://localhost:8000/docs for interactive documentation.

Docker Instructions
1. Build the Docker Image
Navigate to the api/ directory and build the Docker image:

```bash
docker build -t summarization-api .
```
3. Run the Docker Container
Run the container exposing port 8000:
```bash
docker run -p 8000:8000 <imageid>
```

5. Access the API
Access the API at http://localhost:8000/docs for interactive documentation.

Example API Usage
Send a POST request to the /summarize-text endpoint with the following JSON payload:

{
  "text": "This is an example abstract of a research paper...",
  "max_len": 50
}

Response:


{
  "key_idea": "Generated summary of the abstract."
}

Future Work
Extend the API to support multiple models.
Add more robust error handling in the API.
Explore deployment on cloud platforms like AWS or Azure.
License
This project is licensed under the MIT License. ```
