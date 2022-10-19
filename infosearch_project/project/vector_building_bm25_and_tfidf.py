from preprocessing import preprocessing


def vector_building_simple(query, vectorizer, stop_words):
    query_processed = vectorizer.transform([preprocessing(query, stop_words, known_words={})]).toarray()
    return query_processed
