# Домашнее задание 3
Программа состоит из 5 .py файлов:

1. preprocessing.py - функция, отвечающая за предобработку текстов серий сериала, берет на вход строку текста и список стоп-слов;
2. getting_bm25_matrix.py - функция, создающая матрицу с метрикой BM25 на основе предобработанных данных, берет на вход корпус;
3. vector_building.py - функция, получающая на вход запрос и выдающая его вектор (на основе созданной матрицы)
4. getting_similarity.py - функция, считающая близость запроса и матрицы
5. main.py - файл, запускающий программу и объединяющий в себе все функции выше.

Программа для ответа на вопросы, поставленные в домашнем задании, создает файл search_result.txt, в котором перечислены документы в порядке убывания близости к запросу. По умолчанию файлы должны располагаться в одной директории с корпусом, но путь к корпусу можно изменить.

Программа также принимает на вход два файла:
1. data.jsonl (не смог добавить его в репозиторий из-за того, что он слишком тяжелый, а времени в обрез) - файл корпуса вопросов и ответов.
2. all_texts.txt - предобработанный первый файл. Если поставить вместо прописанного в программе названия '', то этот файл создастся с нуля 
