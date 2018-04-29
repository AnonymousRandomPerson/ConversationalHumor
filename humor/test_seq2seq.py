import os
import sys

from utils.file_access import add_seq2seq_module, CONFIG_FOLDER, TEST_OUTPUT_FILE

add_seq2seq_module()

from seq2seq_sub.bin.infer import main

config_string = os.path.join(CONFIG_FOLDER, 'seq2seq_test.yml')

if os.path.exists(TEST_OUTPUT_FILE):
    os.remove(TEST_OUTPUT_FILE)

main(sys.argv, config_string)