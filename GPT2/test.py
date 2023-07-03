import argparse
import os
import sys

import torch
import datetime

from datasets import load_dataset
from transformers import GPT2Tokenizer

MAXLEN_passage = 400
MAXLEN_question = 400

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=32, help='Specify the training batch size')
parser.add_argument('--model_path', type=str, help='Load path of trained model')
parser.add_argument('--predictions_save_path', type=str, help='Load path to which predicted values')

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Choose device
    device = get_default_device()

    test_data = load_dataset('squad_v2', split='validation')

    # Load the GPT tokenizer.
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-large
    # The max model length is 1024 for this model, although the actual embedding size for GPT small is 768
    # The beginning of sequence token <|startoftext|> token has the id 50258
    # The end of sequence token <|endoftext|> has the id 50256
    # The padding token <|pad|> has the id 50257

    count = 0

    model = torch.load(args.model_path, map_location=device)
    model.eval().to(device)

    for ex in test_data:
        if len(ex["answers"]["text"])==0:
            continue
        count+=1
        if count==2:
            break
        print(count)
        print(" ")
        question, passage = ex["question"], ex["context"]
        print(question)
        print(passage)
        print("-------")

        generated = torch.tensor(tokenizer.encode('context:'+ passage + 'question:')).unsqueeze(0)
        generated = generated.to(device)

        sample_outputs = model.generate(
                                generated, 
                                #bos_token_id=random.randint(1,30000),
                                do_sample=True,   
                                top_k=50, 
                                max_length = 300,
                                top_p=0.95, 
                                num_return_sequences=1
                                )

        for i, sample_output in enumerate(sample_outputs):
            print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
