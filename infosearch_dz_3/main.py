import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import json
from nltk.corpus import stopwords
from preprocessing import preprocessing
from getting_bm25_matrix import getting_bm25_matrix
from vector_building import vector_building
from getting_similarity import getting_similarity


def main(file='data.jsonl', file_processed='all_texts.txt'):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
    count_vectorizer = CountVectorizer()
    stop_words = stopwords.words('russian')
    with open(file, 'r', encoding='utf-8') as f:  # Считываем файл
        corpus = list(f)[:50000]

    # Создаем списки названий файлов и их текстов. Я только потом понял, что нужно
    # вытаскивать тексты, как названия, но решил не терять фичу
    data_names = []
    data_texts = []

    for elem in corpus:
        data_names.append(json.loads(elem)['question'])
        max_value = 0
        true_answer = ''
        for answer in json.loads(elem)['answers']:
            if (int(answer['author_rating']['value']) if answer['author_rating']['value'] != '' else 0) > max_value:
                true_answer = answer['text']  # Берем ответ автора с самым большим рейтингом
        data_texts.append(true_answer)

    if file_processed == '':  # Пробовать только в крайнем случае - создание файла занимает около 2 часов!
        texts = []
        known_words = {}
        for index, text in enumerate(data_texts):
            texts.append(
                preprocessing(text, stop_words, known_words))  # Препроцессим тексты и добавляем их в общий список
        with open('all_texts.txt', 'w', encoding='utf-8') as file:
            for line in texts:
                file.write(line)
                file.write('\n')
        file_processed = 'all_texts.txt'

    with open(file_processed, 'r', encoding='utf-8') as file:
        texts_opened = file.readlines()

    bm25 = getting_bm25_matrix(texts_opened, count_vectorizer,
                               tfidf_vectorizer)  # Строим матрицу с посчитанной метрикой BM25
    query = vector_building(input('Введите запрос для поиска: '), count_vectorizer, stop_words)  # Запрашиваем запрос от пользователя
    score = getting_similarity(bm25, query)
    sorted_score_index = np.argsort(score, axis=0)[::-1]
    res = np.array(data_texts)[sorted_score_index.ravel()]  # Производим сортировку

    with open('search_result.txt', 'w', encoding='utf-8') as file:
        for ind, element in enumerate(res):
            file.write(f'{ind + 1} по близости документ - {element}\n')  # Выдача производится через файл search_result.txt

if __name__ == '__main__':
    main()
