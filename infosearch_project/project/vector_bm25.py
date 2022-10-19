from preprocessing import preprocessing


def vector_bm25(query, vectorizer, stop_words):
    query_processed = vectorizer.transform([preprocessing(query, stop_words, known_words={})]).toarray()
    return query_processed
