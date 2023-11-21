import os
import pandas as pd
import random
import math
from collections import OrderedDict, defaultdict
from tqdm import tqdm
import numpy as np
import jsonlines
import re
from matchmaker.autolabel_domain.robust04_nfoldcrosstrain import read_qrels, read_collection, read_candidates

path = '/mnt/c/Users/salthamm/Documents/coding/esci-data/shopping_queries_dataset'
output_dir = "/mnt/c/Users/salthamm/Documents/phd/data/esci/data"

df_examples = pd.read_parquet(os.path.join(path,'shopping_queries_dataset_examples2.parquet'))
df_products = pd.read_parquet(os.path.join(path,'shopping_queries_dataset_products2.parquet'))
df_sources = pd.read_csv(os.path.join(path,"shopping_queries_dataset_sources.csv"))

#merge examples with products
df_examples_products = pd.merge(
    df_examples,
    df_products,
    how='left',
    left_on=['product_locale','product_id'],
    right_on=['product_locale', 'product_id']
)


#filter for task 1

df_task_1 = df_examples_products[df_examples_products["small_version"] == 1]
# only english
df_task_1 = df_task_1[df_task_1["product_locale"] == 'us']
# replace labels

esci_label2gain = {
        'E' : 1.0,
        'S' : 0.1,
        'C' : 0.01,
        'I' : 0.0,
    }

esci_label2relevance_pos = {
        "E" : 3,
        "S" : 1,
        "C" : 2,
        "I" : 0,
    }

rellabel2gain = {
    3: 1.0,
    2: 0.1,
    1: 0.01,
    0: 0.0
}

df_task_1['esci_rellabel'] = df_task_1['esci_label'].apply(lambda esci_label: esci_label2relevance_pos[esci_label])
df_task_1['gain'] = df_task_1['esci_label'].apply(lambda esci_label: esci_label2gain[esci_label])

# cleaning the text
df_task_1['query'] = df_task_1['query'].str.replace("\t"," ").replace("\n"," ").replace("\r"," ").replace('#', '').replace('$','').replace('!','').replace('+','').replace('/','').replace(',','').replace('[','').replace(']','')
df_task_1['product_description'] = df_task_1['product_description'].replace("\t"," ").replace("\n"," ").replace("\r"," ").replace('#', '').replace('$','').replace('!','').replace('+','').replace('/','').replace(',','').replace('[','').replace(']','').replace('</p>', '').replace('<p>', '').replace('<b>', '').replace('</b>', '').replace('<br />', '').replace('<br/>', '').replace('</br>', '').replace('<br>', '').replace('\xa0', '').replace('<li>', '').replace('</li>', '').replace('</ul>', '')
df_task_1['product_title'] = df_task_1['product_title'].replace("\t"," ").replace("\n"," ").replace("\r"," ").replace('#', '').replace('$','').replace('!','').replace('+','').replace('/','').replace(',','').replace('[','').replace(']','').replace('</p>', '').replace('<p>', '').replace('<b>', '').replace('</b>', '').replace('<br />', '').replace('<br/>', '').replace('</br>', '').replace('<br>', '').replace('\xa0', '').replace('<li>', '').replace('</li>', '').replace('</ul>', '')
df_task_1['product_bullet_point'] = df_task_1['product_bullet_point'].replace("\t"," ").replace("\n"," ").replace("\r"," ").replace('#', '').replace('$','').replace('!','').replace('+','').replace('/','').replace(',','').replace('[','').replace(']','').replace('</p>', '').replace('<p>', '').replace('<b>', '').replace('</b>', '').replace('<br />', '').replace('<br/>', '').replace('</br>', '').replace('<br>', '').replace('\xa0', '').replace('<li>', '').replace('</li>', '').replace('</ul>', '')


df_task_1_train = df_task_1[df_task_1["split"] == "train"]
df_task_1_test = df_task_1[df_task_1["split"] == "test"]

list_query_id = df_task_1_train['query_id'].unique()
random.shuffle(list_query_id) #otherwise sorted by alphabetical order
list_query_id_dev = random.sample(list(list_query_id), 888)
list_query_id_train = list(set(list_query_id).difference(set(list_query_id_dev)))

df_train = df_task_1[df_examples_products['query_id'].isin(list_query_id_train)]
df_dev = df_task_1[df_examples_products['query_id'].isin(list_query_id_dev)]

df_task_1_train.to_pickle(os.path.join(output_dir, 'df_train.pkl'))

# now i have train, dev and test

# training
# train_samples.append(InputExample(texts=[row[col_query], row[col_product_title]], label=float(row[col_gain])))
# 'query', 'product_title'

# there are duplicates in the dataset for the queries

# now extract the collection, then the queries for train, dev and test

# first the collection:
# only consider the product_title as text!
collection = dict(zip(df_task_1['product_id'], df_task_1['product_title']))
print(collection.get('B075SCHMPY'))
with open(os.path.join(output_dir, 'collection2.tsv'), 'w') as out:
    for id, text in collection.items():
        text = text.replace("\t"," ").replace("\n"," ").replace("\r"," ").strip()
        id = id.replace("\t"," ").replace("\n"," ").replace("\r"," ").strip()
        out.write(id + '\t' + text + '\n')

queries_test = dict(zip(df_task_1_test['query_id'], df_task_1_test['query']))
print(queries_test.get(12))
with open(os.path.join(output_dir, 'queries_test.tsv'), 'w') as out:
    for id, text in queries_test.items():
        text = text.replace("\t"," ").replace("\n"," ").replace("\r"," ").strip()
        id = str(id).replace("\t"," ").replace("\n"," ").replace("\r"," ").strip()
        out.write(id + '\t' + text + '\n')

queries_dev = dict(zip(df_dev['query_id'], df_dev['query']))
with open(os.path.join(output_dir, 'queries_dev.tsv'), 'w') as out:
    for id, text in queries_dev.items():
        text = text.replace("\t"," ").replace("\n"," ").replace("\r"," ").strip()
        id = str(id).replace("\t"," ").replace("\n"," ").replace("\r"," ").strip()
        out.write(id + '\t' + text + '\n')

queries_train = dict(zip(df_train['query_id'], df_train['query']))
with open(os.path.join(output_dir, 'queries_train.tsv'), 'w') as out:
    for id, text in queries_train.items():
        text = text.replace("\t"," ").replace("\n"," ").replace("\r"," ").strip()
        id = str(id).replace("\t"," ").replace("\n"," ").replace("\r"," ").strip()
        out.write(id + '\t' + text + '\n')

i = 0
# create the qrels for train, dev and test
with open(os.path.join(output_dir, 'qrels/train.qrels.txt'), 'w') as out:
    #df_train = df_train.reset_index()
    for index, row in df_train.iterrows():
        if i < 10:
            print(str(row['query_id']) + ' 0 ' + str(row['product_id']) + ' ' + str(row['esci_rellabel']) + '\n')
        out.write(str(row['query_id']) + ' 0 ' + str(row['product_id']) + ' ' + str(row['esci_rellabel']) + '\n')
        i += 1

with open(os.path.join(output_dir, 'qrels/dev.qrels.txt'), 'w') as out:
    #df_dev = df_dev.reset_index()
    for index, row in df_dev.iterrows():
        out.write(str(row['query_id']) + ' 0 ' + str(row['product_id']) + ' ' + str(row['esci_rellabel']) + '\n')

with open(os.path.join(output_dir, 'qrels/test.qrels.txt'), 'w') as out:
    #df_task_1_test = df_task_1_test.reset_index()
    for index, row in df_task_1_test.iterrows():
        out.write(str(row['query_id']) + ' 0 ' + str(row['product_id']) + ' ' + str(row['esci_rellabel']) + '\n')


# i think the difference is that they train with the gain.... right?
# so the gain should be the score of the positive one and the negative one should be 0

# so create the triples with KD with esci gains!

df_task_1_train = pd.read_pickle(os.path.join(output_dir, 'df_train.pkl'))

# create triples for train, create reranking file for test:
# this done with bm25 search pyserini, for that create psyerini index!

qrels = read_qrels(os.path.join(output_dir, 'qrels/train.qrels.txt'))
queries = read_collection(os.path.join(output_dir, 'queries_train.tsv'))
collection = read_collection(os.path.join(output_dir, 'collection.tsv'))

queries_train = list(queries.keys())
collection_keys = list(collection.keys())
triples = []

stats = defaultdict(int)
n_samples_per_query = -1

if n_samples_per_query > -1:
    print('{} samples per query are added to the training file'.format(n_samples_per_query))
else:
    print('All available samples per query_id are added to the training file')

pos_length = []
neg_length = []
i = 0
for query_id in queries_train:
    if i % 1000 == 0:
        print('finished with {} samples'.format(i))
    rel_dict = qrels.get(query_id)

    # make the positives and negatives the same length, and then zip them!
    if rel_dict:
        if 0 in rel_dict.keys() and 3 in rel_dict.keys():
            neg_list = rel_dict.get(0)
            pos_list = rel_dict.get(3)

            pos_list = [id for id in pos_list if id in collection_keys]
            neg_list = [id for id in neg_list if id in collection_keys]

            pos_length.append(len(pos_list))
            neg_length.append(len(neg_list))

            for pos_id in pos_list:
                neg_id = random.sample(neg_list, 1)[0]

                triples.append((query_id, pos_id, neg_id))
                if i < 10:
                    print(triples[i])
                i += 1

            # neg_listc = neg_list
            # pos_listc = pos_list
            #
            # if len(pos_list) != len(neg_list):
            #     if len(pos_list) > len(neg_list):
            #         stats['pos_larger'] += 1
            #         for i in range(0, math.ceil((len(pos_list) / len(neg_list)) - 1)):
            #             neg_list.extend(neg_listc)
            #         neg_list = neg_list[:len(pos_list)]
            #     else:
            #         stats['neg_larger'] += 1
            #         for i in range(0, math.ceil((len(neg_list) / len(pos_list)) - 1)):
            #             pos_list.extend(pos_listc)
            #         pos_list = pos_list[:len(neg_list)]
            #
            # assert len(pos_list) == len(neg_list)
            #
            # if n_samples_per_query > -1:
            #     if n_samples_per_query <= len(pos_list):
            #         for i in range(n_samples_per_query):
            #             pos_id = pos_list[i]
            #             neg_id = neg_list[i]
            #
            #             triples.append((query_id, pos_id, neg_id))
            #     else:
            #         for i in range(len(pos_list)):
            #             pos_id = pos_list[i]
            #             neg_id = neg_list[i]
            #
            #             triples.append((query_id, pos_id, neg_id))
            # else:
            #     for i in range(len(pos_list)):
            #         pos_id = pos_list[i]
            #         neg_id = neg_list[i]
            #
            #         triples.append((query_id, pos_id, neg_id))

for key, val in stats.items():
    print(f"{key}\t{val}")

print(triples[0])
print(triples[1])

# pos_length = []
# neg_length = []
# for query_id in queries_train:
#
#     rel_dict = qrels.get(query_id)
#
#     # make the positives and negatives the same length, and then zip them!
#     if rel_dict:
#         if 3 in rel_dict.keys():
#             pos_list = rel_dict.get(3)
#
#             pos_list = [id for id in pos_list if id in collection_keys]
#             pos_length.append(len(pos_list))
#
#             neg_list = []
#             mapping = {}
#             if 0 in rel_dict.keys():
#                 neg_list.extend(rel_dict.get(0))
#                 for neg_id in neg_list:
#                     mapping.update({neg_id: rellabel2gain[0]})
#             if 1 in rel_dict.keys():
#                 neg_list.extend(rel_dict.get(1))
#                 for neg_id in neg_list:
#                     mapping.update({neg_id: rellabel2gain[1]})
#             if 2 in rel_dict.keys():
#                 neg_list.extend(rel_dict.get(2))
#                 for neg_id in neg_list:
#                     mapping.update({neg_id: rellabel2gain[2]})
#
#             neg_list = [id for id in neg_list if id in collection_keys]
#             neg_length.append(len(neg_list))
#
#             for pos_id in pos_list:
#                 neg_id = random.sample(neg_list, 1)[0]
#                 triples.append((1, mapping[neg_id] ,query_id, pos_id, neg_id))

print('on average {} positives and {} negatives per query'.format(np.mean(pos_length), np.mean(neg_length)))

# important: shuffle the train data
random.shuffle(triples)

print(triples[0])


with open(os.path.join(output_dir, 'train_triples_nfold2.tsv'), "w", encoding="utf8") as out_file_text, \
        open(os.path.join(output_dir, 'train_triples_nfold_ids2.tsv'), "w", encoding="utf8") as out_file_ids:
    for i, (query_id, pos_id, neg_id) in tqdm(enumerate(triples)):
        try:
            if collection[pos_id].strip() != "" and collection[neg_id].strip() != "":
                out_file_ids.write(str(query_id) + "\t" + pos_id + "\t" + neg_id + "\n")
                out_file_text.write(
                    queries.get(query_id) + "\t" + collection[pos_id] + "\t" + collection[neg_id] + "\n")  #str(pos_score) + '\t' + str(neg_score) + '\t' +
        except:
            print('didnt work for {} and {}'.format(pos_id, neg_id))


# create pyserini index, for that: jsonlines
collection = read_collection(os.path.join(output_dir, 'collection.tsv'))
with jsonlines.open(os.path.join(output_dir, 'collection.jsonl'), mode='w') as writer:
    for id, text in collection.items():
        writer.write({'id': id, 'contents': text})

# then python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator  -threads 1 -input jsonlfile  -index index/ -storePositions -storeDocvectors -storeRaw


#generate qrels from amazon offical file
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


""" 0. Init variables """

col_iteration = "iteration"
col_query_id = "query_id"
col_product_id = "product_id"
col_product_locale = "product_locale"
col_small_version = "small_version"
col_split = "split"
col_esci_label = "esci_label"
col_relevance_pos = "relevance_pos"
col_ranking_postion = "ranking_postion"
col_score = "score"
col_conf = "conf"

max_trec_eval_score = 128
min_trec_eval_score = 0

esci_label2relevance_pos = {
    "E": 3,
    "S": 2,
    "C": 1,
    "I": 0,
}

""" 1. Generate RESULTS file """

locales = [
    "us",
    "es",
    "jp",
]

""" 2. Generate QRELS file """
df_examples = pd.read_parquet(os.path.join(path, 'shopping_queries_dataset_examples.parquet'))
df_products = pd.read_parquet(os.path.join(path, 'shopping_queries_dataset_products.parquet'))
df_examples_products = pd.merge(
    df_examples,
    df_products,
    how='left',
    left_on=[col_product_locale, col_product_id],
    right_on=[col_product_locale, col_product_id]
)
df_examples_products = df_examples_products[df_examples_products[col_small_version] == 1]
df_examples_products = df_examples_products[df_examples_products[col_split] == "test"]

df_examples_products[col_iteration] = 0
df_examples_products[col_relevance_pos] = df_examples_products[col_esci_label].apply(
    lambda esci_label: esci_label2relevance_pos[esci_label])
df_examples_products = df_examples_products[[
    col_query_id,
    col_iteration,
    col_product_id,
    col_relevance_pos,
]]
df_examples_products.to_csv(
    os.path.join(output_dir, "test.qrels.txt"),
    index=False,
    header=False,
    sep=' ',
)





