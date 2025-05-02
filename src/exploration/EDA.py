import pandas as pd
from pandas.core.frame import DataFrame
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.tokenize import word_tokenize


def text_analysis(df: DataFrame) -> None:
    print(f"Total samples: {len(df)}")
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # missing values
    df.dropna(subset=["abstract", "title"], inplace=True)
    print(f"\nTotal samples after dropping NA: {len(df)}")

    # duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"Total samples after dropping duplicates: {len(df)}")

    # length
    df["abstract_length"] = df["abstract"].apply(lambda x: len(str(x).split()))
    df["title_length"] = df["title"].apply(lambda x: len(str(x).split()))

    print("\nDescriptive statistics for text lengths (words):")
    print(df[["abstract_length", "title_length"]].describe())


def get_top_words(text_series: pd.Series, stop_words: list, n=10) -> list:
    # Ensuring all entries are strings
    text_series = text_series.astype(str)
    words = " ".join(text_series).split()
    # We ensure that all characters are alpha and lowercase and we remove stop words
    words = [w.lower() for w in words if w.isalpha() and w.lower() not in stop_words]
    return Counter(words).most_common(n)

def tf_idf_cosine_sim(df: DataFrame) -> np.ndarray:
    tfidf = TfidfVectorizer(stop_words="english")
    # Ensure string type before fitting/transforming
    article_tfidf = tfidf.fit_transform(df["abstract"].astype(str))
    summary_tfidf = tfidf.transform(df["title"].astype(str))

    return cosine_similarity(article_tfidf, summary_tfidf).diagonal()

def jaccard_similarity(a, b):
    try:
      # Basic tokenization and lowercasing
      a_set = set(word_tokenize(str(a).lower()))
      b_set = set(word_tokenize(str(b).lower()))
      intersection = len(a_set.intersection(b_set))
      union = len(a_set.union(b_set))
      return intersection / union if union > 0 else 0
    except Exception as e:
      # Handle potential errors with odd inputs
      print(f"Error calculating Jaccard for: a='{a}', b='{b}'. Error: {e}")
      return 0