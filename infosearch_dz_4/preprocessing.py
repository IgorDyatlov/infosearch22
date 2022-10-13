import pymorphy2
import re


def preprocessing(text, stop_words, known_words):
    morph = pymorphy2.MorphAnalyzer()  # лемматизируем с помощью pymorphy2
    lemmas = []  # сюда кладем все леммы в тексте
    regex_text = re.sub('[^\w\s]', '', text)
    regex_text = re.sub('[\d]', '', regex_text)  # чистим текст от пунктуации, переносов строк и чисел
    regex_text = re.sub('\n', ' ', regex_text)
    for word in regex_text.split(' '):
        if word in stop_words:  # добавляем леммы в словарь со встретившимися словами и в список лемм
            continue
        elif word in known_words:
            lemmas.append(known_words[word])
        else:
            lemma = morph.parse(word.lower())[0].normal_form
            lemmas.append(lemma)
            known_words[word] = lemma
    res = re.sub('  ', ' ', ' '.join(lemmas))  # заменяем двойные пробелы на одинарные и соединяем леммы в строку
    return res
