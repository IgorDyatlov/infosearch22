import streamlit as st
import pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import numpy as np
import json
from vector_bert import vector_bert
from vector_bm25 import vector_bm25
from vector_tfidf import vector_tfidf
# from getting_similarity import getting_similarity
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from similarity import getting_similarity_dot
from vector_building_bm25_and_tfidf import vector_building_simple
from vector_building_bert import vector_building_bert

stop_words = stopwords.words('russian')
# tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
# model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")

with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('bert_indexes.pickle', 'rb') as f:
    bert_indexes = pickle.load(f)
with open('bert_vectorizer.pickle', 'rb') as f:
    bert_vectorizer = pickle.load(f)

with open('tfidf_indexes.pickle', 'rb') as f:
    tfidf_list = pickle.load(f)
    tfidf_indexes = tfidf_list[0]
    tfidf_words = tfidf_list[1]
with open('tfidf_vectorizer.pickle', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('bm25_indexes.pickle', 'rb') as f:
    bm25_indexes = pickle.load(f)
with open('bm25_tfidf_vectorizer.pickle', 'rb') as f:
    bm25_tfidf_vectorizer = pickle.load(f)
with open('bm25_count_vectorizer.pickle', 'rb') as f:
    bm25_count_vectorizer = pickle.load(f)

with open('data_names.txt', 'r', encoding='utf-8') as f:
    data_names = f.readlines()
with open('data_texts.txt', 'r', encoding='utf-8') as f:
    data_texts = f.readlines()
with open('all_texts.txt', 'r', encoding='utf-8') as f:
    all_texts = f.readlines()

query = input('Введите запрос: ')
metric = 'TF-IDF'
if query:
    if metric == 'TF-IDF':
        print(type(vector_building_simple(query, tfidf_vectorizer, stop_words)))
        print(type(tfidf_indexes))
        print(type(bm25_indexes))
        score = getting_similarity_dot(tfidf_indexes, vector_building_simple(query, tfidf_vectorizer, stop_words))
    elif metric == 'BM25':
        score = getting_similarity_dot(bm25_indexes, vector_building_simple(query, bm25_count_vectorizer, stop_words))
    else:
        score = getting_similarity_dot(bert_indexes, vector_building_bert(query, model, tokenizer))
    sorted_score_index = np.argsort(score, axis=0)[::-1]
    print(score.nonzero())
    res = np.array(data_texts)[sorted_score_index.ravel()]