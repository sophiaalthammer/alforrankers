#
# generate a subset for the exercise for the air lecture -> take only documents from bm25 result and given query file
# -------------------------------
#

import argparse
import os
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())

from matchmaker.evaluation.msmarco_eval import *
from matchmaker.utils import parse_candidate_set

from allennlp.common import Params, Tqdm
Tqdm.default_mininterval = 1

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--collection', action='store', dest='collection',
                    help='the full collection file location', required=True)

parser.add_argument('--out-file-subset', action='store', dest='out_file_sub',
                    help='the query output file location', required=True)

parser.add_argument('--qrels', action='store', dest='qrel',
                    help='the qrel file location', required=True)

parser.add_argument('--out-file-qrel', action='store', dest='out_file_qrel',
                    help='qrel subset out', required=True)

parser.add_argument('--bm25', action='store', dest='bm25',
                    help='bm25 output', required=True)

parser.add_argument('--query', action='store', dest='query',
                    help='query.tsv', required=True)

args = parser.parse_args()


#
# load data 
# -------------------------------
# 
bm25_info = parse_candidate_set(args.bm25,1000)

queries = {}
with open(args.query,"r",encoding="utf8") as query_file:
    for line in tqdm(query_file):
        ls = line.split("\t") # id<\t>text ....
        _id = int(ls[0])
        queries[_id] = None #ls[1].rstrip()


docs = {}
for q in queries:
    if q in bm25_info:
        for doc in bm25_info[q]:
            docs[doc] = None

#
# produce output
# -------------------------------
#  
with open(args.out_file_qrel,"w",encoding="utf8") as out_file_qrel:
    with open(args.qrel,"r",encoding="utf8") as qrel:

        for line in tqdm(qrel):
            _id = int(line.split("\t")[0])

            if _id in queries:
                out_file_qrel.write(line)

with open(args.out_file_sub,"w",encoding="utf8") as sub_out_file:
    with open(args.collection,"r",encoding="utf8") as collection:

        for line in tqdm(collection):
            _id = int(line.split("\t")[0])

            if _id in docs:
                sub_out_file.write(line)
