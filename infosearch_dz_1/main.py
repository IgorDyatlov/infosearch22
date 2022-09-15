from indexing_matrix import indexing_matrix
from indexing_dictionary import indexing_dictionary
from preprocessing import preprocessing
import os
import numpy as np
from nltk.corpus import stopwords


def main():
    stop_words = stopwords.words('russian')
    names = ('моника', 'мон', 'рэйчел', 'рейч', 'чендлер', 'чэндлер', 'чен',
             'фиби', 'фибс', 'росс', 'джоуи', 'джои',
             'джо')  # кортеж для того, чтобы имена не стали жертвой чрезмерной лемматизации
    curr_dir = os.getcwd()
    filepath = os.path.join(curr_dir, 'friends-data/')
    dirfiles = os.listdir(filepath)  # получаем путь до наших данных

    texts = []  # список текстов, по которому будем проходиться
    for season in dirfiles:
        season_path = os.path.join(filepath, f'{season}/')
        for script in os.listdir(season_path):
            with open(os.path.join(filepath, f'{season}/{script}'), 'r', encoding='utf-8') as file:
                texts.append(preprocessing(file.read(), stop_words, names))  # каждую серию обрабатываем отдельно, затем
                # складываем обработанный текст в список

    our_matrix = indexing_matrix(texts)  # матрица
    our_dictionary = indexing_dictionary(texts)  # словарь

    with open('matrix_data.txt', 'w', encoding='utf-8') as file:
        max_freq = 0
        for line in np.transpose(our_matrix[0].toarray()):  # транспонируем матрицу и ищем максимальную частотность слов
            # по сумме употреблений в документах
            max_freq = max(max_freq, sum(line))
        max_freq_words = set()
        for index, line in enumerate(np.transpose(our_matrix[0].toarray())):  # снова проходим по матрице и ищем слова с
            # частотностью, равной максимальной
            if sum(line) == max_freq:
                max_freq_words.add(our_matrix[1][index])
        file.write(f'Самое частотное слово - {max_freq_words}. Оно повторяется {max_freq} раз.\n\n')

        min_freq = 1
        for line in np.transpose(our_matrix[0].toarray()):  # проделываем то же самое, что и выше, однако уже для
            # минимальной частотности
            min_freq = min(min_freq, sum(line))
        min_freq_words = set()
        for index, line in enumerate(np.transpose(our_matrix[0].toarray())):
            if sum(line) == min_freq:
                min_freq_words.add(our_matrix[1][index])
        file.write(f'Самые редкие слова: {", ".join(min_freq_words)}. Каждое из них повторяется {min_freq} раз.\n\n')

        words_everywhere = set()  # словарь слов, которые повторяются в каждом документе
        for index, line in enumerate(np.transpose(our_matrix[0].toarray())):  # в транспонированной матрице ищем строки,
            # где есть хотя бы одно повторение слова во всех документах, а затем добавляем их в словарь
            line_set = set(line)
            if 0 not in line_set:
                words_everywhere.add(our_matrix[1][index])
        file.write(f'Слова, которые появляются во всех документах: {", ".join(words_everywhere)}.\n\n')

        pre_names_freq = {}  # словарь всех имен
        for index, line in enumerate(
                np.transpose(our_matrix[0].toarray())):  # транспонируем матрицу и ищем все имена и их частотности
            if our_matrix[1][index] in names:
                pre_names_freq[our_matrix[1][index]] = sum(line)
        names_freq = {
            'моника': pre_names_freq['моника'] + pre_names_freq['мон'],
            'рэйчел': pre_names_freq['рэйчел'] + pre_names_freq['рейч'],
            'чендлер': pre_names_freq['чендлер'] + pre_names_freq['чэндлер'] + pre_names_freq['чен'],
            'фиби': pre_names_freq['фиби'] + pre_names_freq['фибс'],
            'росс': pre_names_freq['росс'],
            'джоуи': pre_names_freq['джоуи'] + pre_names_freq['джои'] + pre_names_freq['джо']
        }  # объединяем некоторые имена для того, чтобы иметь полные, связанные данные (очень плохой способ, но я не смог
        # придумать лучше, если вам тоже не понравится - напишите мне, пожалуйста, как можно сделать лучше
        file.write(
            f'Самый популярный персонаж - {max(names_freq, key=names_freq.get)}, он встречается {names_freq[max(names_freq, key=names_freq.get)]} раз.\n\n')

    with open('dictionary_data.txt', 'w', encoding='utf-8') as file:
        max_freq = 0
        for elem in our_dictionary:  # в словаре ищем самую большую частотность слова
            max_freq = max(sum(our_dictionary[elem].values()), max_freq)
        max_freq_words = set()
        for elem in our_dictionary:  # ищем слово/слова с самой большой частотностью
            if sum(our_dictionary[elem].values()) == max_freq:
                max_freq_words.add(elem)
        file.write(f'Самое частотное слово - {max_freq_words}. Оно повторяется {max_freq} раз.\n\n')

        min_freq = 1
        for elem in our_dictionary:  # в словаре ищем наименьшую частотность слова
            min_freq = min(sum(our_dictionary[elem].values()), min_freq)
        min_freq_words = set()
        for elem in our_dictionary:  # ищем слово/слова с наименьшей частотностью
            if sum(our_dictionary[elem].values()) == min_freq:
                min_freq_words.add(elem)
        file.write(f'Самые редкие слова: {", ".join(min_freq_words)}. Каждое из них повторяется {min_freq} раз.\n\n')

        words_everywhere = set()
        for elem in our_dictionary:  # ищем ключ словаря, значением которого является словарь длиной 165 (165 - количество всех документов)
            if len(our_dictionary[elem]) == 165:
                words_everywhere.add(elem)
        file.write(f'Слова, которые появляются во всех документах: {", ".join(words_everywhere)}.\n\n')

        pre_names_freq = {}
        for elem in our_dictionary:  # ищем в словаре имена и их частотности
            if elem in names:
                for text in our_dictionary[elem]:
                    pre_names_freq[elem] = our_dictionary[elem][text] + pre_names_freq.get(elem, 0)
        names_freq = {
            'моника': pre_names_freq['моника'] + pre_names_freq['мон'],
            'рэйчел': pre_names_freq['рэйчел'] + pre_names_freq['рейч'],
            'чендлер': pre_names_freq['чендлер'] + pre_names_freq['чэндлер'] + pre_names_freq['чен'],
            'фиби': pre_names_freq['фиби'] + pre_names_freq['фибс'],
            'росс': pre_names_freq['росс'],
            'джоуи': pre_names_freq['джоуи'] + pre_names_freq['джои'] + pre_names_freq['джо']
        }  # объединяем некоторые имена для того, чтобы иметь полные, связанные данные
        file.write(
            f'Самый популярный персонаж - {max(names_freq, key=names_freq.get)}, он встречается {names_freq[max(names_freq, key=names_freq.get)]} раз.\n\n')


if __name__ == '__main__':
    main()
