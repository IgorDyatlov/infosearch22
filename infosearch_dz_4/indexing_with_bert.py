import numpy as np
import torch
from mean_pooling import mean_pooling
from sklearn.preprocessing import normalize


def indexing_with_bert(corpus, count_vectorizer, model, tokenizer):
    tf = count_vectorizer.fit_transform(corpus)
    words = count_vectorizer.get_feature_names()
    print(len(words))
    print(len(np.transpose(tf.toarray())))

    def batch(texts,
              n=1):  # Взял со stack overflow реализацию создания батчей - https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
        batches = []
        for ndx in range(0, len(texts), n):
            batches.append(texts[ndx:min(ndx + n, len(texts))])
        return batches

    vectorized = []
    texts_splitted = batch(corpus, n=100)
    for index, batch in enumerate(texts_splitted):
        encoded_input = tokenizer(batch, padding=True, truncation=True, max_length=25, return_tensors='pt')

        with torch.no_grad():
            model_output = model(**encoded_input)

        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        vectorized.append(sentence_embeddings)
        if index % 10 == 0:
            print(len(vectorized))
    return normalize(torch.vstack(vectorized))
