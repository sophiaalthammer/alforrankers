#
# Take .tsv collection files with fulltext and score-id files to produce score-text files
# -------------------------------
#

import argparse
from collections import defaultdict
import os
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())

from matchmaker.evaluation.msmarco_eval import *

from allennlp.common import Params, Tqdm
Tqdm.default_mininterval = 1

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--collection', action='store', dest='collection',
                    help='The full collection file: <did text>', required=True)

parser.add_argument('--query', action='store', dest='query',
                    help='The query text file: <qid text>', required=True)

parser.add_argument('--in-file', action='store', dest='in_file',
                    help='The teacher training score file: <(s_pos s_neg) q_id d_pos_id d_neg_id>', required=True)

parser.add_argument('--add-teacher-scores', action='store', dest='add_teacher',type=bool,
                    help='QA char-span augmented qrels', required=False)

parser.add_argument('--in-file-qa', action='store', dest='in_file_qa',
                    help='QA char-span augmented qrels', required=True)

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='The teacher training score file output (filled with text): <(optional: s_pos s_neg) pos_char_spans q_text d_pos_text d_neg_text>', required=True)


args = parser.parse_args()


#
# load data 
# -------------------------------
# 
queries = {}
with open(args.query,"r",encoding="utf8") as query_file:
    for line in tqdm(query_file):
        ls = line.split("\t") # id<\t>text ....
        queries[ls[0]] = ls[1].rstrip()

docs = {}
with open(args.collection,"r",encoding="utf8") as collection_file:
    for line in tqdm(collection_file):
        ls = line.split("\t") # id<\t>text ....
        docs[ls[0]] = ls[1].rstrip()

qa_qrels = {}
with open(args.in_file_qa,"r",encoding="utf8") as in_file_qa:
    for line in tqdm(in_file_qa):
        ls = line.split("\t") # id<\t>id<t>spans
        qa_qrels[(ls[0],ls[1])] = ls[2].rstrip()

#
# produce output
# -------------------------------
#  
stats = defaultdict(int)
with open(args.out_file,"w",encoding="utf8") as out_file:
    with open(args.in_file,"r",encoding="utf8") as in_file:

        for line in tqdm(in_file):
            line = line.split("\t") # scorpos scoreneg query docpos docneg

            if len(line) == 5:
                q_id = line[2]
                d_pos_id = line[3]
                d_neg_id = line[4].rstrip()
            elif len(line) == 3:
                q_id = line[0]
                d_pos_id = line[1]
                d_neg_id = line[2].rstrip()

            try:
                q_text = queries[q_id]
                doc_pos_text = docs[d_pos_id]
                doc_neg_text = docs[d_neg_id.rstrip()]

                qa_spans = "" 
                if (q_id,d_pos_id) in qa_qrels:
                    qa_spans= qa_qrels[(q_id,d_pos_id)]

                if args.add_teacher:
                    out_file.write(line[0]+"\t"+line[1]+"\t"+qa_spans+"\t"+q_text+"\t"+doc_pos_text+"\t"+doc_neg_text+"\n")
                else:
                    out_file.write(qa_spans+"\t"+q_text+"\t"+doc_pos_text+"\t"+doc_neg_text+"\n")

                stats["success"]+=1
            except KeyError as e:
                stats["key_error"]+=1

for key, val in stats.items():
    print(f"{key}\t{val}")