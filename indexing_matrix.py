from sklearn.feature_extraction.text import CountVectorizer


def indexing_matrix(corpus):
    vectorizer = CountVectorizer(analyzer='word')
    matrix = vectorizer.fit_transform(corpus)  # создаем матрицу с обратным индексом для нашего корпуса
    words = vectorizer.get_feature_names()  # создаем список со всеми словами, который соотносится с индексами в матрице
    return [matrix, words]
