import sys
from get_results import answering_question, process, search2
from model import bert_model 
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-question", type=str, help="question")
# parser.add_argument("-num", type=int, help="number of results to fetch")
args = parser.parse_args()

model_type = 'bert-large-cased-whole-word-masking-finetuned-squad'
model, tokenizer = bert_model(model_type)

question = args.question
num_results = 10
# question = 'How to prepare nitrophosphate' # input('Question: ')
# num_results = args.num

answer = answering_question(model, tokenizer, question, num_results)
answer = answer.capitalize()

print()
print("Question:", question)
print("Answer  :", answer)