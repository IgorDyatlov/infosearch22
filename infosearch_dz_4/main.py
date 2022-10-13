from transformers import AutoTokenizer, AutoModel
import numpy as np
import json
from preprocessing import preprocessing
from indexing_with_bert import indexing_with_bert
from vector_building_bert import vector_building_bert
from getting_similarity import getting_similarity
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


def main(file='data.jsonl', file_processed='all_texts.txt'):
    count_vectorizer = CountVectorizer()
    stop_words = stopwords.words('russian')
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")

    with open(file, 'r', encoding='utf-8') as f:
        corpus = list(f)[:10000]

    data_names = []
    data_texts = []

    for elem in corpus:
        data_names.append(json.loads(elem)['question'])
        max_value = 0
        true_answer = ''
        for answer in json.loads(elem)['answers']:
            if (int(answer['author_rating']['value']) if answer['author_rating']['value'] != '' else 0) > max_value:
                true_answer = answer['text']
        data_texts.append(true_answer)

    if file_processed == '':  # 10000 файлов обрабатываются примерно 20-30 минут
        texts = []
        known_words = {}
        for index, text in enumerate(data_texts):
            texts.append(preprocessing(text, stop_words, known_words))
            if index % 1000 == 0:
                print(index)
                print(len(known_words))
        with open('all_texts.txt', 'w', encoding='utf-8') as file:
            for line in texts:
                file.write(line)
                file.write('\n')
        file_processed = 'all_texts.txt'

    with open(file_processed, 'r', encoding='utf-8') as file:
        texts_opened = file.readlines()

    matrix_bert = indexing_with_bert(data_texts, count_vectorizer, model, tokenizer)
    query = vector_building_bert(input('Введите запрос для поиска: '), model, tokenizer) # Запрашиваем запрос от пользователя
    score = getting_similarity(matrix_bert, query)
    sorted_score_index = np.argsort(score, axis=0)[::-1]
    res = np.array(data_texts)[sorted_score_index.ravel()]  # Производим сортировку

    with open('search_result.txt', 'w', encoding='utf-8') as file:
        for ind, element in enumerate(res):
            file.write(
                f'{ind + 1} по близости документ - {element}\n')  # Выдача производится через файл search_result.txt

if __name__ == '__main__':
    main()