import numpy as np
from nltk.tokenize import sent_tokenize

def preprocess_function(examples, tokenizer, max_input_length, max_target_length):
    # Ensure inputs are strings
    inputs = [str(doc) for doc in examples["abstract"]]
    targets = [str(doc) for doc in examples["title"]]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
    )

    # Setup the tokenizer for targets
    labels = tokenizer(
        text_target=targets,
        max_length=max_target_length,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]

    # Defensive check for invalid token IDs (although less common with AutoTokenizer)
    final_input_ids = []
    for ids in model_inputs["input_ids"]:
        final_input_ids.append([id_ if 0 <= id_ < tokenizer.vocab_size else tokenizer.pad_token_id for id_ in ids])
    model_inputs["input_ids"] = final_input_ids

    final_labels = []
    for ids in model_inputs["labels"]:
        final_labels.append([id_ if 0 <= id_ < tokenizer.vocab_size else tokenizer.pad_token_id for id_ in ids])
    model_inputs["labels"] = final_labels

    return model_inputs

# --- Metrics Computation Function (Common for Seq2Seq) ---
def compute_metrics(eval_pred, tokenizer, rouge_score):
    predictions, labels = eval_pred

    # Ensure numpy arrays
    if isinstance(predictions, tuple): predictions = predictions[0]
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Replace -100 in predictions and labels as we can't decode them
    # Also handles potential out-of-vocab predictions if generation config is loose
    predictions = np.where((predictions < 0) | (predictions >= tokenizer.vocab_size) | (predictions == -100),
                         tokenizer.pad_token_id,
                         predictions)
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    labels = np.where((labels < 0) | (labels >= tokenizer.vocab_size),
                      tokenizer.pad_token_id,
                      labels)

    # Decode generated summaries
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Decode reference summaries
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE expects a newline after each sentence
    # Add simple splitting for rougeLsum calculation
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]

    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract ROUGE f1 scores
    result = {key: value * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}