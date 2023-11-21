import argparse
from tqdm import tqdm
import os

#
# config
#
parser = argparse.ArgumentParser()


parser.add_argument('--out-file', action='store', dest='out_file',
                    help='output folder', required=False,default="/mnt/c/Users/salthamm/Documents/phd/data/beir/cqadupstack" )

args = parser.parse_args()


#
# work
#


import ir_datasets

subsets = ['beir/cqadupstack/android','beir/cqadupstack/english', 'beir/cqadupstack/gaming', 'beir/cqadupstack/gis',
           'beir/cqadupstack/mathematica', 'beir/cqadupstack/physics', 'beir/cqadupstack/programmers',
           'beir/cqadupstack/stats', 'beir/cqadupstack/tex', 'beir/cqadupstack/unix', 'beir/cqadupstack/webmasters',
           'beir/cqadupstack/wordpress']


for subset in subsets:
    out_file_path = os.path.join(args.out_file, subset.split('/')[-1])
    os.mkdir(out_file_path)

    dataset = ir_datasets.load(subset)

    with open(os.path.join(out_file_path,"queries.tsv"),"w",encoding="utf8") as out_file:
     for query in dataset.queries_iter():
        #query # namedtuple<query_id, text>
        out_file.write(query.query_id + "\t" +query.text.replace("\t"," ").replace("\n"," ").strip()+"\n")

    with open(os.path.join(out_file_path,"collection.tsv"),"w",encoding="utf8") as out_file:
     for doc in dataset.docs_iter():
        #doc # namedtuple<doc_id, title, text>
        text = doc.title.replace("\t", " ").replace("\n", " ").strip() + " " + doc.text.replace("\t", " ").replace("\n",
                                                                                                                   " ").strip()
        out_file.write(doc.doc_id + "\t" + text.strip()[:100_000] + "\n")

    with open(os.path.join(out_file_path,"qrels.txt"),"w",encoding="utf8") as out_file:
     for qrel in dataset.qrels_iter():
        qrel # namedtuple<query_id, doc_id, relevance, iteration>
        out_file.write(qrel.query_id + " Q0 " +qrel.doc_id+" "+str(qrel.relevance)+"\n")