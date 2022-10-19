import streamlit as st
import pickle
import xlsxwriter
from io import BytesIO
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from similarity import getting_similarity_dot
from vector_building_bm25_and_tfidf import vector_building_simple
from vector_building_bert import vector_building_bert

stop_words = stopwords.words("russian")

with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)
with open("model.pickle", "rb") as f:
    model = pickle.load(f)

with open("bert_indexes.pickle", "rb") as f:
    bert_indexes = pickle.load(f)
with open("bert_vectorizer.pickle", "rb") as f:
    bert_vectorizer = pickle.load(f)

with open("tfidf_indexes.pickle", "rb") as f:
    tfidf_list = pickle.load(f)
    tfidf_indexes = tfidf_list[0]
    tfidf_words = tfidf_list[1]
with open("tfidf_vectorizer.pickle", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("bm25_indexes.pickle", "rb") as f:
    bm25_indexes = pickle.load(f)
with open("bm25_tfidf_vectorizer.pickle", "rb") as f:
    bm25_tfidf_vectorizer = pickle.load(f)
with open("bm25_count_vectorizer.pickle", "rb") as f:
    bm25_count_vectorizer = pickle.load(f)

with open("data_names.txt", "r", encoding="utf-8") as f:
    data_names = f.readlines()
with open("data_texts.txt", "r", encoding="utf-8") as f:
    data_texts = f.readlines()
with open("all_texts.txt", "r", encoding="utf-8") as f:
    all_texts = f.readlines()


# Запуск - streamlit run C:/Users/User/PycharmProjects/infosearch_project/main.py

def blank_line(num_of_lines):
    for _ in range(num_of_lines):
        st.write("")


def st_to_excel(df,
                encoding_parameter):  # Функцию взял отсюда - https://discuss.streamlit.io/t/download-button-for-csv-or-xlsx-file/17385/2
    output_bytes = BytesIO()
    writer = pd.ExcelWriter(output_bytes, engine='xlsxwriter')
    df.to_excel(writer, encoding=encoding_parameter, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'})
    worksheet.set_column('A:A', None, format1)
    writer.save()
    processed_data = output_bytes.getvalue()
    return processed_data


st.set_page_config(
    layout="wide",
    page_title="Поискови4ок",
    page_icon="♟",
    initial_sidebar_state="expanded"
)

st.header('', anchor='up')
st.markdown("<h1 style='text-align: center; color: LightCoral;'>Мэйл в поисках любви</h1>", unsafe_allow_html=True)

first_col, second_col, third_col = st.columns([10, 80, 10])

metric = st.sidebar.selectbox("Выберите метрику поиска", ["TF-IDF", "BM25", "BERT"])
ceiling = st.sidebar.number_input(
    "Количество ответов",
    min_value=1,
    max_value=50000,
    help="""Задайте количество ответов, которое будет выдано (от 1 до 50000)""",
)
user_encoding = st.sidebar.selectbox(
    "Выберите кодировку выдачи", ["utf-8", "windows-1251"],
    help="Задайте кодировку, в которой будет скачиваться выдача (не влияет на просмотр выдачи на сайте)"
)

with second_col:
    blank_line(5)
    with st.expander("ℹ️ - О поисковике", expanded=False):
        st.write(
            """
    Данный поисковик предоставляет возможность осуществлять поиск по 50000 ответам из датасета Ответов Мэйл.Ру с помощью трех разных метрик - TF-IDF, BM25 и BERT.\n 
    По умолчанию поиск производится по метрике TF-IDF, выдается самый близкий к запросу ответ, а выдача скачивается в кодировке UTF-8. Для смены метрики, количества выдаваемых ответов и кодировки воспользуйтесь сайдбаром слева.
            """
        )
    blank_line(1)
    query = st.text_input("Введите запрос")  # Есть любовный вопрос? Вбей запрос!
    blank_line(1)
    if query:
        if metric == "TF-IDF":
            score = getting_similarity_dot(tfidf_indexes, vector_building_simple(query, tfidf_vectorizer, stop_words))
        elif metric == "BM25":
            score = getting_similarity_dot(bm25_indexes,
                                           vector_building_simple(query, bm25_count_vectorizer, stop_words))
        else:
            score = getting_similarity_dot(bert_indexes, vector_building_bert(query, model, tokenizer))
        sorted_score_index = np.argsort(score, axis=0)[::-1]
        res = np.array(data_texts)[sorted_score_index.ravel()]
        output_dict = {
            "Позиция по релевантности": [],
            "Ответ": []
        }
        for ind in range(ceiling):
            output_dict["Позиция по релевантности"].append(ind + 1)
            output_dict["Ответ"].append(res[ind])
        output = pd.DataFrame(data=output_dict)
        output_xlsx = st_to_excel(output, user_encoding)
        st.table(output)
        st.markdown("[Вверх](#up)")

with first_col:
    blank_line(10)
    if query:
        st.download_button(
            label='Скачать выдачу в CSV',
            data=output.to_csv(encoding=user_encoding),
            file_name='output.csv'
        )

with third_col:
    blank_line(10)
    if query:
        st.download_button(
            label='Скачать выдачу в XLSX',
            data=output_xlsx,
            file_name='output.xlsx'
        )
