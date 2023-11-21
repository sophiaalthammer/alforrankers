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


args = parser.parse_args()


out_folder = os.path.join(args.out_folder,"msmarco-passage-v2")
os.makedirs(out_folder,exist_ok=True)

dataset = ir_datasets.load("msmarco-passage-v2")
dataset_docs = ir_datasets.load("msmarco-document-v2").docs_store()

#
# get collection 
#


print("Get collection")
collection_path = os.path.join(out_folder,"augmented-title-collection.tsv")
with open(collection_path,"w",encoding="utf8") as out_file:
 for doc in tqdm(dataset.docs_iter()):

    doc_infos = dataset_docs.get(doc.msmarco_document_id)

    cat_text = doc_infos.title + " " + doc.text

    out_file.write(doc.doc_id + "\t" +' '.join(cat_text.replace("\t"," ").replace("\n"," ").replace("\r"," ").strip().split())+"\n")
