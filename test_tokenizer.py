
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

tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b', trust_remote_code=True)
# prompt='cpu_svctm:4.12,Incoming_network_traffic:3.64,Outgoing_network_traffic:4.44,Used_inodes:1.13,'
prompt='12cpu,npu'
answer='53,23,3,2,12'
# [17577, 24, 27397, 12636, 12, 10, 7, 16, 23, 6, 44865, 24, 8852, 24, 40257, 12, 13, 7, 21, 16, 6, 62303, 24, 8852, 24, 40257, 12, 16, 7, 16, 16, 6, 21935, 24, 27627, 106, 12, 9, 7, 9, 13, 6, 130001, 130004, 5, 15, 13, 6, 10, 13, 6, 13, 6, 10, 6, 9, 10, 130005]
# [17577, 24, 27397, 12636, 12, 16, 7, 9, 10, 6, 44865, 24, 8852, 24, 40257, 12, 13, 7, 21, 16, 6, 62303, 24, 8852, 24, 40257, 12, 16, 7, 16, 16, 6, 21935, 24, 27627, 106, 12, 9, 7, 9, 13, 6, 130001, 130004, 5, 15, 13, 6, 10, 13, 6, 13, 6, 10, 6, 9, 10, 130005]
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