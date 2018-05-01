import argparse
import os

from utils.file_access import add_module, CONFIG_FOLDER, DATA_FOLDER, SUBWORD_MODULE, TEST_OUTPUT_FILE

add_module(SUBWORD_MODULE)

import subword_nmt.apply_bpe as apply_bpe

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")

    parser.add_argument('--input', '-i', required=True, help="Input file.")
    parser.add_argument('--codes', '-c', required=True, help="File with BPE codes.")
    parser.add_argument('--vocabulary', '-v', default=None,
        help="Vocabulary file. If provided, this script reverts any merge operations that produce an OOV.")
    parser.add_argument(
        '--vocabulary-threshold', '-t', type=int, default=None,
        help="Vocabulary threshold. If vocabulary is provided, any word with frequency < threshold will be treated as OOV")
    args = parser.parse_args()

    for file_suffix in ('_sources', '_targets'):
        root, ext = os.path.splitext(args.input)
        input_file = root + file_suffix + ext
        output_file = root + '_bpe' + file_suffix + ext

        args_list = []
        args_list.extend(['-i', os.path.join(DATA_FOLDER, input_file)])
        args_list.extend(['-o', os.path.join(DATA_FOLDER, output_file)])
        args_list.extend(['-c', os.path.join(DATA_FOLDER, args.codes)])
        if args.vocabulary:
            args_list.extend(['--vocabulary', os.path.join(DATA_FOLDER, args.vocabulary)])
        if args.vocabulary_threshold:
            args_list.extend(['--vocabulary-threshold', args.vocabulary_threshold])

        apply_bpe.main(args_list)

if __name__ == '__main__':
    main()