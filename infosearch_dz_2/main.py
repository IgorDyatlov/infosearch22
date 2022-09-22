import numpy as np
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import preprocessing
from vector_building import vector_building
from getting_similarity import getting_similarity
from indexing_matrix import indexing_matrix

def main(filepath='friends-data'):
    stop_words = stopwords.words('russian')
    vectorizer = TfidfVectorizer(analyzer='word')
    names = ('моника', 'мон', 'рэйчел', 'рейч', 'чендлер', 'чэндлер', 'чен',
                 'фиби', 'фибс', 'росс', 'джоуи', 'джои',
                 'джо')  # кортеж для того, чтобы имена не стали жертвой чрезмерной лемматизации

    dirfiles = os.listdir(filepath)  # парсим путь

    texts = []
    for season in dirfiles:
        season_path = os.path.join(filepath, f'{season}/')
        for script in os.listdir(season_path):
            with open(os.path.join(filepath, f'{season}/{script}'), 'r', encoding='utf-8') as file:
                texts.append(preprocessing(file.read(), stop_words, names))

    indexed_matrix = indexing_matrix(texts, vectorizer)  # создаем матрицу
    query = input('Введите запрос для поиска: ')  # запрашиваем запрос от пользователя

    docs = {}
    similarities = getting_similarity(np.array(indexed_matrix[0].toarray()), np.transpose(vector_building(query, stop_words, vectorizer, names)[0].toarray()))  # здесь мы одновременно высчитываем и вектор запроса, и получаем косинусную близость между запросом и матрицей
    file_num = 0
    for season in dirfiles:
        season_path = os.path.join(filepath, f'{season}/')
        for script in os.listdir(season_path):
            docs[script] = similarities[file_num]  # строим словарь с ключом в виде названия файла и со значением в виде косинусной близости
            file_num += 1

    sorted_dict = {}
    sorted_keys = sorted(docs, key=docs.get, reverse=True)
    for w in sorted_keys:
        sorted_dict[w] = docs[w]  # сортируем словарь по убыванию в значении

    with open('search_result.txt', 'w', encoding='utf-8') as file:
        for index, key in enumerate(sorted_dict):
            file.write(f'{index + 1} по близости документ - {key}\n')  # выдача производится через файл search_result.txt

if __name__ == '__main__':
    main()
