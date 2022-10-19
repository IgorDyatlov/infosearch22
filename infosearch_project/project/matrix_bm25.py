from scipy import sparse

def matrix_bm25(corpus, count_vectorizer, tfidf_vectorizer):
    tf = count_vectorizer.fit_transform(corpus)
    tfidf = tfidf_vectorizer.fit_transform(corpus)
    idf = tfidf_vectorizer.idf_

    len_d = tf.sum(axis=1)
    avdl = len_d.mean()

    k = 2
    b = 0.75
    values = []
    rows = []
    cols = []
    B_1 = (k * (1 - b + b * len_d / avdl))

    for i, j in zip(*tf.nonzero()):
        values.append(tf[i, j] * idf[j] * (k + 1))
        rows.append(i)
        cols.append(j)
    A = sparse.csr_matrix((values, (rows, cols)))

    values_bm = []
    rows_bm = []
    cols_bm = []
    for i, j in zip(*tf.nonzero()):
        values_bm.append(float((A[i, j] / (tf[i, j] + B_1[i])).tolist()[0][0]))
        rows_bm.append(i)
        cols_bm.append(j)
    return sparse.csr_matrix((values_bm, (rows_bm, cols_bm)))
