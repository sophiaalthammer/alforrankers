#
# reduce ranked file by randomly sampling 10% from top 100-1000, keep all top100
# -------------------------------

import random
random.seed(42)

import argparse
import os
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())

from collections import defaultdict

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='training output text file location', required=True)

parser.add_argument('--in-file', action='store', dest='in_file',
                    help='ranked output file', required=True)

args = parser.parse_args()

max_triples = 10_000_000
max_doc_char_length = 150_000

max_doc_token_length = 10000


max_rank_to_use = 100
subsample_cutoff = 40

#
# produce output
# -------------------------------
#  

triples = []

stats = defaultdict(int)

with open(args.in_file,"r",encoding="utf8") as candidate_file,\
     open(args.out_file,"w",encoding="utf8") as out_file:

    for line in tqdm(candidate_file):
        #if random.random() <= 0.5: continue #skip some entries for faster processing
        ls = line.split()
        if len(ls) == 6:
            [topicid, _ , unjudged_docid, rank, _ , _ ] = ls
        else:
            [topicid, unjudged_docid, rank, _ ] = ls

        if int(rank) > max_rank_to_use: 
            continue

        if int(rank) <= subsample_cutoff:
            #if random.random() < 0.7: continue # skip 70% of candidates to speed up things...
            #else:
            stats['< '+str(subsample_cutoff)+' sampling count'] += 1
        else:
            if random.random() <= 0.9: continue # skip 90% of candidates assumong top1k -> same number of samples from 0-100 as 101 - 1000
            else:
                stats['> '+str(subsample_cutoff)+' sampling count'] += 1

        out_file.write(line)

for key, val in stats.items():
    print(f"{key}\t{val}")