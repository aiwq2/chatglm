
import logging
import os
import sys
import json

import numpy as np
from datasets import load_dataset
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from trainer_seq2seq import Seq2SeqTrainer

tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b', trust_remote_code=True)
prompt='上涨指标及其对应的值为:Disk_svctm:2.48,Incoming_network_traffic:3.64,Outgoing_network_traffic:4.44,Used_inodes:1.13,'
answer='53,23,3,2,12'

a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

if len(a_ids) > 511:
    a_ids = a_ids[:511]

if len(b_ids) > 100:
    b_ids = b_ids[: 100]

input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

context_length = input_ids.index(tokenizer.bos_token_id)
mask_position = context_length - 1
labels = [-100] * context_length + input_ids[mask_position+1:]

# pad_len = 612 - len(input_ids)
# input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
# labels = labels + [tokenizer.pad_token_id] * pad_len
print(f'input_ids:{input_ids}')
print(f'labels:{labels}')