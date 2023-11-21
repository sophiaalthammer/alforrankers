import os
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import pickle

def load_file(path):
    collection = {}
    with open(path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            splitted = line.split('\t')
            collection.update({splitted[0]: splitted[1].rstrip('\n')})
    return collection


# path to the queries
path = '/mnt/c/Users/salthamm/Documents/phd/data/clinical-trials/data/queries_2021.tsv' #queries_2022.tsv

kw_model = KeyBERT(model=SentenceTransformer('dmis-lab/biobert-base-cased-v1.1'))
queries = load_file(path)

#for query_id, query_text in queries.items():
keywords = kw_model.extract_keywords(list(queries.values()), top_n=25)
keys = []
for i in keywords:
    keys.append([j[0] for j in i])
    
#queries["keywords"] = [" ".join(i) for i in keys]

#with open(os.path.join('/'.join(path.split('/')[:-1]), 'queries_2022_keywords.pkl'), 'wb') as out_file2:
#    pickle.dump(keywords, out_file2)


with open(os.path.join('/'.join(path.split('/')[:-1]), 'queries_2021_with_keywords25.tsv'), 'w') as out_file:
    i = 0
    for query_id, query_text in queries.items():
        out_file.write(query_id + '\t' + ' '.join([key[0] for key in keywords[i]]) + ' ' + query_text + '\n')
        i += 1

with open(os.path.join('/'.join(path.split('/')[:-1]), 'queries_2021_only_keywords25.tsv'), 'w') as out_file:
    i = 0
    for query_id, query_text in queries.items():
        out_file.write(query_id + '\t' + ' '.join([key[0] for key in keywords[i]])+ '\n')
        i += 1






