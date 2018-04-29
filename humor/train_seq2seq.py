import argparse
import os
import sys

from utils.file_access import add_module, CONFIG_FOLDER, SEQ2SEQ_MODULE

add_module(SEQ2SEQ_MODULE)

import seq2seq_sub.bin.train as train

def main():
    parser = argparse.ArgumentParser(description='Trains a seq2seq model.')
    parser.add_argument('-c', '--config-file', required=True, help='The config file to use for the seq2seq training.')
    args = parser.parse_args()

    example_config_path = os.path.join(SEQ2SEQ_MODULE, 'example_configs')

    config_names = ['nmt_small.yml', 'train_seq2seq.yml', 'text_metrics_bpe.yml']
    config_paths = [os.path.join(example_config_path, config_name) for config_name in config_names]
    config_paths.append(os.path.join(CONFIG_FOLDER, args.config_file))

    # Make a comma-separated list of config paths to be compatible with the seq2seq input.
    first = True
    config_string = ''
    for config_path in config_paths:
        if not first:
            config_string += ','
        config_string += config_path
        first = False

    sys.argv = sys.argv[:1]
    train.main(sys.argv, config_string)

if __name__ == '__main__':
    main()