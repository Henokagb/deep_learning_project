from fastapi import FastAPI, Body
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline


tokenizer = AutoTokenizer.from_pretrained("Mug3n24/bart-base-finetune-finetuned-research-papers-XX")
model = AutoModelForSeq2SeqLM.from_pretrained("Mug3n24/bart-base-finetune-finetuned-research-papers-XX")
summarizer = pipeline("text2text-generation", model="Mug3n24/bart-base-finetune-finetuned-research-papers-XX")

app = FastAPI()


def get_summary(text, max_len):
    if summarizer is None:
        return "Summarizer pipeline not available."

    try:
        generated_summary = summarizer(text, max_length=max_len, min_length=5, do_sample=False)[0]['generated_text']
    except Exception as e:
        print(f"Error during summarization pipeline execution: {e}")
        generated_summary = "[Error generating summary]"

    return f"{generated_summary}"


@app.post("/summarize-text")
async def summarize_text(
    text: str = Body(..., embed=True), max_len: int = Body(..., embed=True)
):
    if max_len < 5:
        max_len = 5
    summary = get_summary(text, max_len)
    return {"key_idea": summary}
