import argparse
import os
import sys
import json

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import random
import time
import datetime

from datasets import load_dataset
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, ElectraConfig
from transformers import get_linear_schedule_with_warmup

def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
device = get_default_device()

inputs_text = "hello"
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
text = "hello"
encoding_dict = tokenizer(text)
configuration = GPT2Config.from_pretrained('gpt2-large', output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained("gpt2-large", config=configuration)
model.resize_token_embeddings(len(tokenizer))
model.to(device)
outputs = model(**encoding_dict)
print(outputs.loss, outputs.logits)
