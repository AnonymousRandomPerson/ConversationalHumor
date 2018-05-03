import argparse
import os

from utils.file_access import add_corpus_argument, add_output_suffix_argument, BPE_PREFIX, DATA_FOLDER, TEXT_EXTENSION, VOCAB_PREFIX

import subword_nmt.apply_bpe as apply_bpe

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Learn BPE-based word segmentation')

    add_corpus_argument(parser)
    add_output_suffix_argument(parser)
    parser.add_argument(
        '-t', '--vocabulary-threshold', type=int, default=None,
        help='Vocabulary threshold. If vocabulary is provided, any word with frequency < threshold will be treated as OOV')
    args, _ = parser.parse_known_args()

    for file_suffix in ('sources', 'targets'):
        file_suffix = '_' + file_suffix
        root, ext = os.path.splitext(args.corpus_file)
        input_file = root + file_suffix + ext
        output_file = root + '_' + BPE_PREFIX + file_suffix + ext

        args_list = []
        args_list.extend(['-i', os.path.join(DATA_FOLDER, input_file)])
        args_list.extend(['-o', os.path.join(DATA_FOLDER, output_file)])
        args_list.extend(['-c', os.path.join(DATA_FOLDER, BPE_PREFIX + '_' + args.output_suffix + TEXT_EXTENSION)])
        #args_list.extend(['--vocabulary', os.path.join(DATA_FOLDER, VOCAB_PREFIX + '_' + args.output_suffix + TEXT_EXTENSION)])
        if args.vocabulary_threshold:
            args_list.extend(['--vocabulary-threshold', args.vocabulary_threshold])

        apply_bpe.main(args_list)

if __name__ == '__main__':
    main()