from nltk.tokenize import sent_tokenize


def lead_3_summary(text: str) -> str:
    try:
        return "\n".join(sent_tokenize(text)[:3])
    except Exception as e:
        print(f"Error tokenizing text for Lead-3: {e}")
        # Fallback: return first N words if sentence tokenization fails
        return ' '.join(str(text).split()[:50]) # Arbitrary fallback length

def evaluate_baseline(dataset, metric):
    # Ensure 'abstract' and 'title' columns exist and are strings
    abstracts = [str(text) for text in dataset["abstract"]]
    titles = [str(text) for text in dataset["title"]]

    summaries = [lead_3_summary(text) for text in abstracts]

    # ROUGE expects newline separated sentences
    # The lead_3_summary function already does this
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in titles]

    return metric.compute(predictions=summaries, references=decoded_labels, use_stemmer=True)