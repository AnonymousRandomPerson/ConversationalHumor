import os

from utils.file_access import add_module, CONFIG_FOLDER, DATA_FOLDER, SUBWORD_MODULE, TEST_OUTPUT_FILE

add_module(SUBWORD_MODULE)

from subword_nmt.learn_joint_bpe_and_vocab import main

input_file = 'twitter_raw.txt'
mid_file = 'bpe_raw.txt'
output_file = 'vocab_raw.txt'
vocab_size = 100000

args_list = []
args_list.extend(['-i', os.path.join(DATA_FOLDER, input_file)])
args_list.extend(['-o', os.path.join(DATA_FOLDER, mid_file)])
args_list.extend(['--write-vocabulary', os.path.join(DATA_FOLDER, output_file)])
args_list.extend(['-s', str(vocab_size)])

main(args_list)