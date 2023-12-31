#
# generate train.triples.tsv tuples from candidate set files 
# & original train triple ids -> now we add bm25 to the output (thats why we are doing this)
# -------------------------------
#

import argparse
import os
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())

from allennlp.common import Params, Tqdm
Tqdm.default_mininterval = 1

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='training output file location', required=True)

parser.add_argument('--candidate-file', action='store', dest='candidate_file',
                    help='trec ranking file location (lucene output)', required=True)

parser.add_argument('--collection-file', action='store', dest='collection_file',
                    help='collection.tsv location', required=True)

parser.add_argument('--triple-ids', action='store', dest='triples_file',
                    help='qid 2x pid triples', required=True)

parser.add_argument('--query-file', action='store', dest='query_file',
                    help='query.tsv location', required=True)

args = parser.parse_args()


#
# load data 
# -------------------------------
# 
collection = {} # int id -> full line dictionary
with open(args.collection_file,"r",encoding="utf8") as collection_file:
    for line in tqdm(collection_file):
        ls = line.split("\t") # id<\t>text ....
        _id = int(ls[0])
        collection[_id] = ls[1].rstrip()

queries = {}
with open(args.query_file,"r",encoding="utf8") as query_file:
    for line in tqdm(query_file):
        ls = line.split("\t") # id<\t>text ....
        _id = int(ls[0])
        queries[_id] = ls[1].rstrip()

scores = {} # qid -> pid
with open(args.candidate_file,"r",encoding="utf8") as candidate_file:
    for line in tqdm(candidate_file):
        ls = line.split() # 2 Q0 1782337 1 21.656799 Anserini

        query_id = int(ls[0])
        doc_id = int(ls[2])

        if query_id not in scores:
            scores[query_id]={}

        scores[query_id][doc_id] = ls[4]
        
#
# produce output
# -------------------------------
#  
#max_rank = args.top_n

skipped_count=0

with open(args.out_file,"w",encoding="utf8") as out_file:
    with open(args.triples_file,"r",encoding="utf8") as triples_file:

        for line in tqdm(triples_file):
            ls = line.split() # qid pid1 pid2

            query_id = int(ls[0])
            doc_id1 = int(ls[1])
            doc_id2 = int(ls[2])
            
            if query_id in scores and doc_id1 in scores[query_id] and doc_id2 in scores[query_id]:
                out_file.write("\t".join([queries[query_id],collection[doc_id1],collection[doc_id2],scores[query_id][doc_id1],scores[query_id][doc_id2]])+"\n")
            else:
                skipped_count+=1

print("Finished! skipped:",skipped_count)