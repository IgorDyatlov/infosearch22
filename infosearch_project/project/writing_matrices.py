import json
import pickle
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import preprocessing
from nltk.corpus import stopwords
from matrix_bert import matrix_bert
from matrix_bm25 import matrix_bm25
from matrix_tfidf import matrix_tfidf


def creating_matrices(file='data.jsonl', file_processed='all_texts.txt'):
    tfidf_vectorizer = TfidfVectorizer(analyzer='word')
    bm25_count_vectorizer = CountVectorizer()
    bm25_tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
    bert_vectorizer = CountVectorizer()
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    stop_words = stopwords.words('russian')

    with open('tokenizer.pickle', 'wb') as f:
        pickle.dump(tokenizer, f)
    with open('model.pickle', 'wb') as f:
        pickle.dump(model, f)

    with open(file, 'r', encoding='utf-8') as f:  # Считываем файл
        corpus = list(f)[:50000]

    # Создаем списки названий файлов и их текстов. Я только потом понял, что нужно
    # вытаскивать тексты, как названия, но решил не терять фичу
    data_names = []
    data_texts = []

    print('CREATING DATA FILES...')

    for elem in corpus:
        data_names.append(json.loads(elem)['question'])
        max_value = 0
        true_answer = ''
        for answer in json.loads(elem)['answers']:
            if (int(answer['author_rating']['value']) if answer['author_rating']['value'] != '' else 0) > max_value:
                true_answer = answer['text']  # Берем ответ автора с самым большим рейтингом
        data_texts.append(true_answer)

    with open('data_names.txt', 'w', encoding='utf-8') as file:
        for elem in data_names:
            file.write(f'{elem}\n')

    print('DATA_NAMES DONE')

    with open('data_texts.txt', 'w', encoding='utf-8') as file:
        for elem in data_texts:
            file.write(f'{elem}\n')

    print('DATA_TEXTS DONE')

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

    print('ALL TEXTS HAVE BEEN PROCESSED')

    with open(file_processed, 'r', encoding='utf-8') as file:
        texts_opened = file.readlines()

    print('BUILDING MATRICES...')

    bert_indexes = matrix_bert(data_texts, bert_vectorizer, model, tokenizer)
    with open('bert_indexes.pickle', 'wb') as file:
        pickle.dump(bert_indexes, file)
    with open('bert_vectorizer.pickle', 'wb') as file:
        pickle.dump(bert_vectorizer, file)
    print('BERT DONE')

    bm25_indexes = matrix_bm25(texts_opened, bm25_count_vectorizer, bm25_tfidf_vectorizer)
    with open('bm25_indexes.pickle', 'wb') as file:
        pickle.dump(bm25_indexes, file)
    with open('bm25_count_vectorizer.pickle', 'wb') as file:
        pickle.dump(bm25_count_vectorizer, file)
    with open('bm25_tfidf_vectorizer.pickle', 'wb') as file:
        pickle.dump(bm25_tfidf_vectorizer, file)
    print('BM25 DONE')

    tfidf_indexes = matrix_tfidf(texts_opened, tfidf_vectorizer)
    with open('tfidf_indexes.pickle', 'wb') as file:
        pickle.dump(tfidf_indexes, file)
    with open('tfidf_vectorizer.pickle', 'wb') as file:
        pickle.dump(tfidf_vectorizer, file)
    print('TFIDF DONE')


if __name__ == '__main__':
    creating_matrices()
