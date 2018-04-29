import argparse
import os
import sys

from utils.file_access import add_module, CONFIG_FOLDER, SEQ2SEQ_MODULE, TEST_OUTPUT_FILE

add_module(SEQ2SEQ_MODULE)

import seq2seq_sub.bin.infer as infer

def main():
    parser = argparse.ArgumentParser(description='Tests a seq2seq model.')
    parser.add_argument('-c', '--config-file', required=True, help='The config file to use for the seq2seq testing.')
    args = parser.parse_args()

    config_string = os.path.join(CONFIG_FOLDER, args.config_file)

    if os.path.exists(TEST_OUTPUT_FILE):
        os.remove(TEST_OUTPUT_FILE)

    sys.argv = sys.argv[:1]
    infer.main(sys.argv, config_string)

if __name__ == '__main__':
    main()