import os
import sys

from utils.file_access import CONFIG_FOLDER

seq2seq_module = 'seq2seq_sub'

sys.path.append(seq2seq_module)

from seq2seq_sub.bin.train import main

example_config_path = os.path.join(seq2seq_module, 'example_configs')

config_names = ['nmt_small.yml', 'train_seq2seq.yml', 'text_metrics_bpe.yml']
config_paths = [os.path.join(example_config_path, config_name) for config_name in config_names]
config_paths.append(os.path.join(CONFIG_FOLDER, 'seq2seq_train.yml'))

# Make a comma-separated list of config paths to be compatible with the seq2seq input.
first = True
config_string = ''
for config_path in config_paths:
    if not first:
        config_string += ','
    config_string += config_path
    first = False

main(sys.argv, config_string)