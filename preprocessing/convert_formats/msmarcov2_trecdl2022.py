import argparse
from tqdm import tqdm
import os


#
# work
#

out_file_path = "/mnt/c/Users/salthamm/Documents/phd/data/msmarco-passage-v2/data"

import ir_datasets
# dataset = ir_datasets.load("msmarco-passage-v2/trec-dl-2021/judged")
#
# with open(os.path.join(out_file_path,"trec_2021_queries_judged.tsv"),"w",encoding="utf8") as out_file:
#     for query in dataset.queries_iter():
#         # namedtuple<query_id, text>
#         out_file.write(query.query_id + "\t" +query.text.replace("\t"," ").replace("\n"," ").strip()+"\n")
#
# with open(os.path.join(out_file_path,"trec_2021_qrels_judged.txt"),"w",encoding="utf8") as out_file:
#     for qrel in dataset.qrels_iter():
#         out_file.write(qrel.query_id + " Q0 " + qrel.doc_id + " " + str(qrel.relevance) + "\n")
#         # namedtuple<query_id, doc_id, relevance, iteration>


dataset = ir_datasets.load("msmarco-passage-v2/trec-dl-2022")

with open(os.path.join(out_file_path,"trec_2022_queries.tsv"),"w",encoding="utf8") as out_file:
    for query in dataset.queries_iter():
        # namedtuple<query_id, text>
        out_file.write(query.query_id + "\t" +query.text.replace("\t"," ").replace("\n"," ").strip()+"\n")

