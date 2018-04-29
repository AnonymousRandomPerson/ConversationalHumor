import argparse
import os

from utils.file_access import open_data_file

def main(args_list: list = None):
    parser = argparse.ArgumentParser(description='Delete a saved model')
    parser.add_argument('-c', '--corpus-file', required=True, help='The name of the corpus to get data from.')
    args = parser.parse_args(args_list)

    with open_data_file(args.corpus_file) as corpus_file:
        lines = corpus_file.readlines()

    root, ext = os.path.splitext(args.corpus_file)

    with open_data_file(root + '_sources' + ext, 'w') as source_file:
        with open_data_file(root + '_targets' + ext, 'w') as target_file:
            i = 0
            for line in lines:
                if i == 0:
                    source_file.write(line)
                elif i == 1:
                    target_file.write(line)
                i += 1
                if i >= 3:
                    i = 0

if __name__ == '__main__':
    main()