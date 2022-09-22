from preprocessing import preprocessing


def vector_building(query, stop_words, vectorizer, names):
    query_processed = vectorizer.transform(preprocessing(query, stop_words, names).split(' '))  # строим вектор, предварительно очищая и лемматизируя запрос
    return query_processed
