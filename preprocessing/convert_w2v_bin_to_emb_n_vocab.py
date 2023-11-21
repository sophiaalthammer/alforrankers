#
# create a pre-trained embedding with gensim w2v - using the output of tokenize_files.py (spacy tokenizer)
# -------------------------------

import argparse
import os
import sys
sys.path.append(os.getcwd())

import gensim
from gensim.models import KeyedVectors


#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--out-file-vocab', action='store', dest='out_file_vocab',
                    help='output file', required=True)

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='output file', required=True)

parser.add_argument('--in-file', action='store', dest='in_file',
                    help='input file', required=True)

args = parser.parse_args()


model = KeyedVectors.load_word2vec_format(args.in_file,binary=True)
model.save_word2vec_format(args.out_file,binary=False)

with open(args.out_file_vocab,encoding="utf8",mode="w") as out:
    out.write("@@UNKNOWN@@\n")
    for v in model.key_to_index.keys():
        out.write(v+"\n")
