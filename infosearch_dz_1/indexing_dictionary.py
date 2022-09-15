from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def indexing_dictionary(corpus):
    dictionary = {}  # основной словарь
    vectorizer = CountVectorizer(analyzer='word')
    matrix = vectorizer.fit_transform(corpus)  # создаем матрицу на основе нашего корпуса
    words = vectorizer.get_feature_names()  # создаем список слов, соотносящийся с индексами в матрице
    for elem in range(len(np.transpose(matrix.toarray()))):  # транспонируем матрицу, чтобы проходиться по слову, а не по тексту
        docs = {}  # словарь с метаинформацией и обратной индексацией
        for index, num in enumerate(np.transpose(matrix.toarray())[elem]):  # из матрицы достаем индекс текста и количество употреблений слова в тексте
            if int(num) != 0:
                docs[index] = int(num)  # добавляем эту информацию в словарь
        dictionary[words[elem]] = docs  # создаем пару слово - словарь с метаинформацией и обратной индексацией в нашем основном словаре
    return dictionary
