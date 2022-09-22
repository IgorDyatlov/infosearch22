import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def getting_similarity(matrix, vector):
    return cosine_similarity(matrix, np.transpose(vector))  # вычисляем косинусную близость
