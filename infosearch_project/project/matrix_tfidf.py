def matrix_tfidf(corpus, vectorizer):  # строим матрицу
    matrix = vectorizer.fit_transform(corpus)
    words = vectorizer.get_feature_names()
    return [matrix, words]
