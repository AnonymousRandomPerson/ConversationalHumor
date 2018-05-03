import argparse
import os

from utils.file_access import add_corpus_argument, add_output_suffix_argument, BPE_PREFIX, DATA_FOLDER, TEXT_EXTENSION, VOCAB_PREFIX

import subword_nmt.learn_joint_bpe_and_vocab as learn_joint_bpe_and_vocab

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Learn BPE-based word segmentation')

    add_corpus_argument(parser)
    add_output_suffix_argument(parser)
    parser.add_argument('-s', '--symbols', type=int, default=10000,
                        help='Create this many new symbols (each representing a character n-gram) (default: %(default)s))')

    args, _ = parser.parse_known_args()

    args_list = []
    args_list.extend(['-i', os.path.join(DATA_FOLDER, args.corpus_file)])
    args_list.extend(['-o', os.path.join(DATA_FOLDER, BPE_PREFIX + '_' + args.output_suffix + TEXT_EXTENSION)])
    args_list.extend(['--write-vocabulary', os.path.join(DATA_FOLDER, VOCAB_PREFIX + '_' + args.output_suffix + TEXT_EXTENSION)])
    args_list.extend(['-s', str(args.symbols)])

    learn_joint_bpe_and_vocab.main(args_list)

if __name__ == '__main__':
    main()