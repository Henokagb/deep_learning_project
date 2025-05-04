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
*   **`results/`**: The place to store outputs like saved model checkpoints.
*   **`requirements.txt`**: Generated using `pip freeze > requirements.txt`. Ensures others can recreate the exact environment.

## Running the project

The notebook contains the differents steps of realising constructing the model with some descriptions with Markdown.
First you need to install the required modules:
```bash
pip install -r requirements.txt
```

The notebook uses some functions that we coded in separate files for modularization.You need to have the `src` folder with all its files on the server on which the notebook is executed.
The dataset is in the `data` project.


## Run the API Locally
Navigate to the api/ directory and run the FastAPI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
Once the server is running, you can access the API at http://localhost:8000/docs for interactive testing.

Example API Usage:
Send a POST request to the /summarize-text endpoint with the following JSON payload:

```bash
{
  "text": "This is an example abstract of a research paper...",
  "max_len": 50
}
```

Response:
```bash
{
  "key_idea": "Generated summary of the abstract."
}
```

## Build the Docker Image
If you want to get an image of the API:
Navigate to the api/ directory and build the Docker image:

```bash
docker build -t summarization-api .
```
Run the container exposing port 8000:
```bash
docker run -p 8000:8000 <imageid>
```
Access the API at http://localhost:8000/docs.

## Deployment on render

We also deployed the API on the platform Render from its Docker image.
Since we used a free instance, it spins down with inactivity, which can delay requests by 50 seconds or more.
It is available there: https://summarize-api-x7cu.onrender.com/docs

