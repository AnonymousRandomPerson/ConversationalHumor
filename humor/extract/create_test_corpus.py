import argparse
import random

from extract import split_sets
from utils.file_access import open_data_file

corpus_name = 'twitter_filtered.txt'
save_file = 'twitter_test.txt'
num_examples = 10

def run() -> None:
    """
    Runs the program.
    """
    examples = []
    with open_data_file(corpus_name) as corpus_file:
        i = 0
        current_line = ''
        for line in corpus_file:
            current_line += line
            i += 1
            if i >= 3:
                i = 0
                examples.append(current_line)
                current_line = ''

    num_examples_cap = min(num_examples, len(examples))

    example_indices = random.sample(range(len(examples)), num_examples_cap)
    save_examples = [examples[i] for i in example_indices]
    with open_data_file(save_file, 'w') as save:
        for example in save_examples:
            save.write(example)

    split_sets.main(['-c', save_file])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract a subset of a corpus into another file')
    parser.add_argument('-c', '--corpus-file', help='The name of the corpus to get data from.')
    parser.add_argument('-n', '--num-examples', type=int, help='The number of examples to retrieve from the corpus.')
    parser.add_argument('-s', '--save-file', help='The name of the file to save data to.')
    args = parser.parse_args()

    if args.corpus_file:
        corpus_name = args.corpus_file
    if args.num_examples:
        num_examples = args.num_examples
    if args.save_file:
        save_file = args.save_file

    run()