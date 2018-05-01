import argparse
import os

from utils.file_access import add_module, CONFIG_FOLDER, DATA_FOLDER, SUBWORD_MODULE, TEST_OUTPUT_FILE

add_module(SUBWORD_MODULE)

import subword_nmt.learn_joint_bpe_and_vocab as learn_joint_bpe_and_vocab

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")

    parser.add_argument('--input', '-i', required=True, help="Input texts")
    parser.add_argument('--output', '-o', required=True, help="Output file for BPE codes.")
    parser.add_argument('--symbols', '-s', type=int, default=10000,
        help="Create this many new symbols (each representing a character n-gram) (default: %(default)s))")
    parser.add_argument('--write-vocabulary', '-w', required=True, dest='vocab',
        help='Write to these vocabulary files after applying BPE. One per input text. Used for filtering in apply_bpe.py')

    args = parser.parse_args()

    args_list = []
    args_list.extend(['-i', os.path.join(DATA_FOLDER, args.input)])
    args_list.extend(['-o', os.path.join(DATA_FOLDER, args.output)])
    args_list.extend(['--write-vocabulary', os.path.join(DATA_FOLDER, args.vocab)])
    args_list.extend(['-s', str(args.symbols)])

    learn_joint_bpe_and_vocab.main(args_list)

if __name__ == '__main__':
    main()