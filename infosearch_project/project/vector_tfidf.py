from preprocessing import preprocessing


def vector_tfidf(query, stop_words, vectorizer):
    query_processed = vectorizer.transform(preprocessing(query, stop_words, known_words={}).split(' '))  # строим вектор, предварительно очищая и лемматизируя запрос
    return query_processed
