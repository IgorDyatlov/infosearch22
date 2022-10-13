import torch
from mean_pooling import mean_pooling


def vector_building_bert(query, model, tokenizer):
    encoded_input = tokenizer(query, padding=True, truncation=True, max_length=25, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings
