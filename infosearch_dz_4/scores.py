from indexing_with_bert import indexing_with_bert
from vector_building_bert import vector_building_bert
from getting_bm25_matrix import getting_bm25_matrix
from vector_building_bm25 import vector_building_bm25
from getting_similarity import getting_similarity
import numpy as np


def comparing_scores(data_texts, data_names, texts_opened, stop_words, count_vectorizer, tfidf_vectorizer, model, tokenizer):
    answers_bert = indexing_with_bert(data_texts, count_vectorizer, model, tokenizer)
    questions_bert = vector_building_bert(data_names, count_vectorizer, tokenizer)
    full_score = 0
    for index, question in enumerate(questions_bert):
        score = getting_similarity(answers_bert, question)
        sorted_score_indx = np.argsort(score, axis=0)[::-1]
        sorted_res = np.array(data_texts)[sorted_score_indx.ravel()]
        if data_texts[index] in sorted_res.tolist()[:5]:
            full_score += 1
    res = full_score / 10000

    answers_bm25 = getting_bm25_matrix(texts_opened, count_vectorizer, tfidf_vectorizer)
    questions_bm25 = vector_building_bm25(data_names, count_vectorizer, stop_words)
    full_score_bm = 0
    for index, question in enumerate(questions_bm25):
        score = getting_similarity(answers_bm25, question)
        sorted_score_indx = np.argsort(score, axis=0)[::-1]
        sorted_res = np.array(data_texts)[sorted_score_indx.ravel()]
        if data_texts[index] in sorted_res.tolist()[:5]:
            full_score_bm += 1
    res_bm = full_score_bm / 10000

    return f'Качество на топ-5 у BERT - {res}\nКачество на топ-5 у BM25 - {res_bm}'
