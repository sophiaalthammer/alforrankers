#
# generate triples for a ranked input file
# -------------------------------
#

import argparse
from collections import defaultdict
import os
import random
import sys
from typing import Counter
sys.path.append(os.getcwd())

from allennlp.common import Tqdm
Tqdm.default_mininterval = 1


import numpy as np
import statistics

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--in-rankings', action='store', dest='in_file',
                    help='tsv collection file', required=True)

parser.add_argument('--out-triple-ids', action='store', dest='out_file',
                    help='tsv collection file', required=True)


args = parser.parse_args()


#
# load data & match & output
# -------------------------------
#  

num_out_triples = 50_000_000

min_pos_score = 4
max_diff_to_be_pos = 0.5
min_diff_to_neg = 4

pos_by_qid = defaultdict(list)
neg_by_qid = defaultdict(list)
stats = defaultdict(int)

with open(args.in_file, "r", encoding="utf8") as qf:
    current_q_id = ""
    current_top_score = 0
    for line in qf:
        ls = line.split() 
        if current_q_id != ls[0]:
            current_q_id = ls[0]
            current_top_score = float(ls[3])
            if current_top_score >= min_pos_score:
                pos_by_qid[ls[0]].append((ls[1],float(ls[3])))
                stats["total_pos"]+=1
        else:
            score = float(ls[3])
            if score >= current_top_score - max_diff_to_be_pos and score >= min_pos_score:
                pos_by_qid[ls[0]].append((ls[1],score))
                stats["total_pos"]+=1
            elif score < current_top_score - min_diff_to_neg:
                if ls[0] in pos_by_qid:
                    neg_by_qid[ls[0]].append((ls[1],score))
                    stats["total_neg"]+=1

query_ids = sorted(list(set(pos_by_qid.keys()).intersection(set(neg_by_qid.keys()))))
stats["total_queries"] = len(query_ids)

for key, val in stats.items():
    print(f"{key}\t{val}")

already_output_triples = set()

with open(args.out_file,"w",encoding="utf8") as out_file:
    while len(already_output_triples) < num_out_triples:

        q_id = random.choice(query_ids)

        pos = random.choice(pos_by_qid[q_id])
        neg = random.choice(neg_by_qid[q_id])

        triple = (q_id,pos[0],neg[0])

        if triple not in already_output_triples:
            already_output_triples.add(triple)

            # pos_score<t>neg_score<t>query_id<t>pos_id<t>neg_id
            out_file.write(f"{pos[1]} {neg[1]} {q_id} {pos[0]} {neg[0]}\n")

            stats["output_triples"] += 1
        else:
            stats["blocked_duplicates"] += 1

for key, val in stats.items():
    print(f"{key}\t{val}")
