import argparse
import os
import sys
import json
import time
import openai
import random
from datasets import load_dataset

parser = argparse.ArgumentParser(description='Get all command line arguments.')
#parser.add_argument('--subset_path', type=str, help='Load path of test data')
parser.add_argument('--api_key', type=str, help='Load OpenAI API key')
parser.add_argument('--save_path', type=str, help='Load path to save results with increased distractors')


def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')


    openai.api_key = args.api_key

    with open(args.save_path) as f:
        expanded_examples = json.load(f)

    start_point = len(expanded_examples)

    sample_data = []
    batch_examples = []

    train_data = load_dataset('squad_v2', split='train[-5%:]')
    random.seed(8888)
    random_indices = [random.randint(0,len(train_data)) for i in range(1000)]
    count = 0
    for ind in random_indices:
        if len(train_data[ind]["answers"]["text"])==0:
            continue
        count+=1
        if count==20:
            break
        passage = train_data[ind]["context"]
    sample_data.append(passage)

    '''with open(args.subset_path) as f:
        sample_data = json.load(f)'''

    for count, context in enumerate(sample_data):
        if count < start_point: continue
        prompt = "Generate three questions for the following context. One easy question, one medium question and one hard question\n" + context + "\nPut the generated questions into a json file with keys 'easy','medium' and 'hard'"
        model = "gpt-3.5-turbo"
        response = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}])
        questions = response.choices[0].message.content.strip()
        questions_s = questions.split('"')
        easy = questions_s[3]
        medium = questions_s[7]
        hard = questions_s[11]
        curr_example = {'context': context, 'easy':easy , 'medium': medium, 'hard': hard}
        batch_examples.append(curr_example)
        if len(batch_examples) == 1:
            expanded_examples += batch_examples
            batch_examples = []
            with open(args.save_path, 'w') as f:
                json.dump(expanded_examples, f)
            print("Saved up to:", count)
            print("----------------------")


if __name__ == '__main__':
    args = parser.parse_args()

    for count in range(1,20):
        try:
            main(args)
            time.sleep(1)
            break
        except openai.error.RateLimitError:
            print("openai.error.RateLimitError... #{}".format(count))
            print("restart in 10 seconds")
            time.sleep(10)
        except openai.error.ServiceUnavailableError:
            print("openai.error.ServiceUnavailableError... #{}".format(count))
            print("restart in 10 seconds")
            time.sleep(10)
