#
# tokenize a file with spacy tokenizer -> so that we don't have to do it on the fly
# -------------------------------
#
# usage:
# python matchmaker/preprocessing/tokenize_files.py --in-file <path> --out-file <path> --reader-type <labeled_tuple or triple>

import argparse
import os
import sys
sys.path.append(os.getcwd())
from tqdm import tqdm

from matchmaker.dataloaders.ir_labeled_tuple_loader import *
from matchmaker.dataloaders.ir_tuple_loader import *
from matchmaker.dataloaders.ir_triple_loader import *
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.common import Params, Tqdm
Tqdm.default_mininterval = 1

from hunspell import Hunspell
hunsp = Hunspell('en_US-large', hunspell_data_dir='/home/shofstaetter/hunspell')

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='output file', required=True)

parser.add_argument('--in-file', action='store', dest='in_file',
                    help='input file', required=True)

parser.add_argument('--reader-type', action='store', dest='reader_type',
                    help='labeled_tuple or triple or labeled_single', required=True)

parser.add_argument('--output-type', action='store', dest='output_type',
                    help='same or text_only (only used for labeled_single)', required=False)


args = parser.parse_args()


#
# load data (tokenize) & write out lines
# -------------------------------
#  

if args.reader_type=="labeled_tuple":
    loader = IrLabeledTupleDatasetReader(lazy=True,tokenizer=WordTokenizer()) # explicit spacy tokenize
elif args.reader_type=="labeled_single":
    loader = IrTupleDatasetReader(lazy=True,target_tokenizer=WordTokenizer()) # explicit spacy tokenize
elif args.reader_type=="triple":
    loader = IrTripleDatasetReader(lazy=True,tokenizer=WordTokenizer()) # explicit spacy tokenize
else:
    raise Exception("wrong reader_type:" + args.reader_type)


stat_total_words = 0
stat_total_wrong_words = 0
stat_total_replaced_words = 0

def spellcheck(word):
    global stat_total_words, stat_total_wrong_words, stat_total_replaced_words

    stat_total_words += 1
    if len(word) < 4:
        return word

    if hunsp.spell(word):
        return word

    stat_total_wrong_words += 1

    sug = hunsp.suggest(word)
    if len(sug) == 0:
        return word

    stat_total_replaced_words+= 1
    return sug[0]



with open(args.out_file,"w",encoding="utf8") as out_file:
    instances = loader.read(args.in_file)
    for i in tqdm(instances):
        if args.reader_type=="labeled_tuple":

            # query_id, doc_id, query_sequence, doc_sequence
            out_file.write("\t".join([
                str(i["query_id"].label),
                str(i["doc_id"].label),
                " ".join(spellcheck(t.text) for t in i["query_tokens"]),
                " ".join(spellcheck(t.text) for t in i["doc_tokens"])])+"\n")

        elif args.reader_type=="triple":
            # query_sequence, doc_pos_sequence, doc_neg_sequence
            out_file.write("\t".join([
                " ".join(spellcheck(t.text) for t in i["query_tokens"].tokens),
                " ".join(spellcheck(t.text) for t in i["doc_pos_tokens"].tokens),
                " ".join(spellcheck(t.text) for t in i["doc_neg_tokens"].tokens)])+"\n")

        elif args.reader_type=="labeled_single":
            # source, target 
            if args.output_type == "same":
                out_file.write(i["source_tokens"].tokens[0].text +"\t"+" ".join(spellcheck(t.text) for t in i["target_tokens"].tokens)+"\n")
            else:
                out_file.write(" ".join(spellcheck(t.text.lower()) for t in i["target_tokens"].tokens)+"\n")


print("stat_total_words", stat_total_words)
print("stat_total_wrong_words", stat_total_wrong_words, "(",stat_total_wrong_words/stat_total_words,")")
print("stat_total_replaced_words", stat_total_replaced_words,"(",stat_total_replaced_words/stat_total_wrong_words," of wrong)")