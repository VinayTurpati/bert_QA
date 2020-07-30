import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

def bert_model(model_type):
    model = BertForQuestionAnswering.from_pretrained(model_type)
    tokenizer = BertTokenizer.from_pretrained(model_type)
    return model, tokenizer