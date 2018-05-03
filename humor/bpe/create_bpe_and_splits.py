import sys

import bpe.convert_bpe as convert_bpe
import bpe.create_bpe_vocab as create_bpe_vocab
import extract.split_sets as split_sets

# -c <corpus-name> -o <output-suffix> -s <vocab-size>

split_sets.main(sys.argv)
create_bpe_vocab.main()
convert_bpe.main()