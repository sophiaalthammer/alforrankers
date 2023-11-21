#
# output the saved secondary information & plaintext for given query, doc pairs
#

import argparse
import os
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.common import Params, Tqdm
Tqdm.default_mininterval = 1
import numpy
from matchmaker.dataloaders.bling_fire_tokenizer import *
tokenizer = BlingFireTokenizer()
import statistics

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--in-file', action='store', dest='in_file',
                    help='', required=True)

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='', required=False)

args = parser.parse_args()

with open(args.in_file,"r",encoding="utf8") as in_file,\
     open(args.out_file,"w",encoding="utf8") as out_file:
    for line in tqdm(in_file):
        ls = line.split("\t") # qid<\t>did<\t>query<\t>doc ....
        
        doc_sequence = text_to_sentences(ls[3][:8*2000]).split("\n")
        
        for i,sent in enumerate(doc_sequence):
            out_file.write(str(ls[0])+"\t"+str(ls[1])+"_"+str(i)+"\t"+ls[2]+"\t"+sent+"\n")
