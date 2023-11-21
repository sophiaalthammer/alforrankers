import argparse
from tqdm import tqdm
import os
import ir_datasets

#
# config
#
parser = argparse.ArgumentParser()


parser.add_argument('--out-folder', action='store', dest='out_folder',
                    help='output folder', required=False,default="C:\\Users\\sebas\\data\\ir-data-test")

parser.add_argument('--dataset', action='store', dest='dataset',
                    help='output folder', required=False,default="msmarco-passage-v2")


args = parser.parse_args()


out_folder = os.path.join(args.out_folder,args.dataset)
os.makedirs(out_folder,exist_ok=True)

query_sets_by_collection = {
    "msmarco-document": [("orcas",1)],
    
    "msmarco-passage-v2": [("dev1",1),("dev2",1)],
    "msmarco-document-v2": [("dev1",1),("dev2",1)],
    "natural-questions": [("train",1),("dev",1)],
    "antique": [("train",1),("test",1)],
    
    "beir/climate-fever": [("",1)],
    "beir/trec-covid": [("",1)],
    "beir/webis-touche2020": [("",1)],
    "beir/hotpotqa": [("train",1),("dev",1),("test",1)],
    "beir/fiqa": [("train",1),("dev",1),("test",1)],
    "beir/dbpedia-entity": [("dev",1),("test",1)],

    "beir/nfcorpus":[("train",1),("dev",1),("test",1)],
    "beir/webis-touche2020/v2":[("",1)],
    "cranfield":[("",1)],
}

document_concatenator_by_collection = {
    "msmarco-document": lambda doc_tup: doc_tup.title + " " + doc_tup.body,

    "msmarco-passage-v2": lambda doc_tup: doc_tup.text,
    "msmarco-document-v2": lambda doc_tup: doc_tup.title + " " + doc_tup.body,
    "natural-questions": lambda doc_tup: doc_tup.document_title + " " + doc_tup.text,
    "antique": lambda doc_tup: doc_tup.text,
    "beir/climate-fever": lambda doc_tup: doc_tup.title + " " + doc_tup.text,
    "beir/trec-covid": lambda doc_tup: doc_tup.title + " " + doc_tup.text,
    "beir/webis-touche2020": lambda doc_tup: doc_tup.title + " " + doc_tup.text,
    "beir/hotpotqa": lambda doc_tup: doc_tup.title + " " + doc_tup.text,
    "beir/fiqa": lambda doc_tup: doc_tup.title + " " + doc_tup.text,
    "beir/dbpedia-entity": lambda doc_tup: doc_tup.title + " " + doc_tup.text,
    "beir/nfcorpus":lambda doc_tup: doc_tup.title + " " + doc_tup.text,
    "beir/webis-touche2020/v2":lambda doc_tup: doc_tup.title + " " + doc_tup.text,
    "cranfield":lambda doc_tup: doc_tup.text,

}

relevance_mapper_by_collection = {
    "msmarco-document": lambda rel: rel,

    "msmarco-passage-v2": lambda rel: rel,
    "msmarco-document-v2": lambda rel: rel,
    "natural-questions": lambda rel: rel,
    "antique": lambda rel: rel - 1,
    "beir/climate-fever": lambda rel: rel,
    "beir/trec-covid": lambda rel: rel,
    "beir/webis-touche2020": lambda rel: rel,
    "beir/hotpotqa": lambda rel: rel,
    "beir/fiqa": lambda rel: rel,
    "beir/dbpedia-entity": lambda rel: rel,
    "beir/nfcorpus":lambda rel: rel,
    "beir/webis-touche2020/v2":lambda rel: rel,
    "cranfield":lambda rel: max(0,rel),

}


dataset = ir_datasets.load(args.dataset)

#
# get collection 
#

print("Get collection")
collection_path = os.path.join(out_folder,"collection.tsv")
with open(collection_path,"w",encoding="utf8") as out_file:
 for doc in tqdm(dataset.docs_iter()):
    text = ' '.join((document_concatenator_by_collection[args.dataset](doc)).replace("\t"," ").replace("\n"," ").replace("\r"," ").strip().split())
    if text.strip() != "": # very few. but on some collections there are empty documents -> which breaks downstream code that depends on 2 values in the tsv file -> while stripping whitespace ends
        out_file.write(doc.doc_id + "\t" + text + "\n")
    else:
        print("Empty text for doc_id: " + doc.doc_id)
#
# queries
#

query_config = []

for query_set,binarization in query_sets_by_collection[args.dataset]:
    if query_set == "":
        print("Get all queries for: "+args.dataset)
        dataset = ir_datasets.load(args.dataset)
        query_set = "all"
    else:
        print("Get queries for: "+query_set)
        dataset = ir_datasets.load(args.dataset+"/"+query_set)

    query_text_path = os.path.join(out_folder,query_set+"_queries.tsv")
    with open(query_text_path,"w",encoding="utf8") as out_file:
        for query in dataset.queries_iter():
           out_file.write(query.query_id + "\t" +query.text.replace("\t"," ").replace("\n"," ").strip()+"\n")


    qrel_path = os.path.join(out_folder,query_set+"_qrels.txt")
    with open(qrel_path,"w",encoding="utf8") as out_file:
        for qrel in dataset.qrels_iter():
            # namedtuple<query_id, doc_id, relevance, iteration>
            out_file.write(qrel.query_id + " Q0 " +qrel.doc_id+" "+str(relevance_mapper_by_collection[args.dataset](qrel.relevance))+"\n")

    query_config.append("  " + query_set+ "_top1000:" + \
                        "\n    queries_tsv: " + query_text_path + \
                        "\n    qrels: " + qrel_path + \
                        "\n    binarization_point: "+str(binarization) + \
                        "\n    top_n: 1000")

#
# create config
#

config = "collection_tsv: " + collection_path + "\nquery_sets:\n" + "\n".join(query_config)

conf_path = os.path.join(out_folder,"config.yaml")
with open(conf_path,"w",encoding="utf8") as out_file:
    out_file.write(config)

print("Config for "+args.dataset+" dataset:")
print(config)
