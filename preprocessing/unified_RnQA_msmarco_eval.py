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


parser.add_argument('--in-file', action='store', dest='in_file',
                    help='The eval-re-ranking file: <q_id d_id q_text d_text>', required=True)

parser.add_argument('--in-file-qa', action='store', dest='in_file_qa',
                    help='QA char-span augmented qrels', required=True)

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='The eval-re-ranking file: <q_id d_id q_text d_text qa_span>', required=True)

parser.add_argument('--out-file-answers', action='store', dest='out_file_answers',
                    help='The eval-answer file: <q_id a_text a2_text a3_text ...>', required=True)


args = parser.parse_args()


#
# load data 
# -------------------------------
# 

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
qid_answers = defaultdict(list)
with open(args.out_file,"w",encoding="utf8") as out_file:
    with open(args.in_file,"r",encoding="utf8") as in_file:

        for line in tqdm(in_file):
            ls = line.split("\t") # q_id d_id q_text d_text

            try:
                qa_spans = "" 
                if (ls[0],ls[1]) in qa_qrels:
                    qa_spans= qa_qrels[(ls[0],ls[1])]
                    stats["added_qas"]+=1
                    spans = qa_spans.split()
                    for span in spans:
                        span = span.split(",")
                        qid_answers[ls[0]].append(ls[3][int(span[0]):int(span[1])])

                out_file.write(line.strip()+"\t"+qa_spans+"\n")

                stats["total"]+=1
            except KeyError as e:
                stats["key_error"]+=1

with open(args.out_file_answers,"w",encoding="utf8") as out_file:
    for q,answ in qid_answers.items():
        out_file.write(q+"\t"+"\t".join(answ)+"\n")

for key, val in stats.items():
    print(f"{key}\t{val}")