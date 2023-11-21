import os
import argparse
import random
import math
import copy
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())


def read_qrels(qrels_file_path):
    qrels = {}
    with open(qrels_file_path, 'r') as qrels_file:
        lines = qrels_file.readlines()

        for line in lines:
            line = line.split(' ')
            query_id = line[0]
            doc_id = line[2]
            rel_grade = int(line[3])
            if qrels.get(query_id):
                if qrels.get(query_id).get(rel_grade):
                    qrels.get(query_id).get(rel_grade).append(doc_id)
                else:
                    qrels.get(query_id).update({rel_grade: [doc_id]})
            else:
                qrels.update({query_id: {rel_grade: [doc_id]}})
    return qrels


def create_train_data(queries_train, queries, collection, qrels_file_path, out_file):
    qrels = read_qrels(qrels_file_path)
    collection_keys = list(collection.keys())
    triples = []
    for query_id in queries_train:

        rel_dict = qrels.get(query_id)

        # make the positives and negatives the same length, and then zip them!
        if rel_dict:
            if 0 in rel_dict.keys() and 1 in rel_dict.keys():
                pos_list = rel_dict.get(1)
                neg_list = rel_dict.get(0)

                if 2 in rel_dict.keys():
                    pos_list.append(rel_dict.get(2))
                    print('also use 2 as rel grade')

                print('length of pos {} and neg {}'.format(len(pos_list), len(neg_list)))

                pos_list = [id for id in pos_list if id in collection_keys]
                neg_list = [id for id in neg_list if id in collection_keys]

                print('length of pos {} and neg {}'.format(len(pos_list), len(neg_list)))

                neg_listc = neg_list.copy()
                pos_listc = pos_list.copy()

                if len(pos_list) != len(neg_list):
                    if len(pos_list) > len(neg_list):
                        print('pos larger')
                        for i in range(0, math.ceil((len(pos_list) / len(neg_list)) - 1)):
                            neg_list.extend(neg_listc)
                        neg_list = neg_list[:len(pos_list)]
                    else:
                        print('neg larger')
                        for i in range(0, math.ceil((len(neg_list) / len(pos_list)) - 1)):
                            pos_list.extend(pos_listc)
                        pos_list = pos_list[:len(neg_list)]

                assert len(pos_list) == len(neg_list)

                for i in range(len(pos_list)):
                    pos_id = pos_list[i]
                    neg_id = neg_list[i]

                    triples.append((query_id, pos_id, neg_id))

    # important: shuffle the train data
    random.shuffle(triples)

    with open(os.path.join(out_file, 'train_triples.tsv'), "w", encoding="utf8") as out_file_text, \
            open(os.path.join(out_file, 'train_triples_ids.tsv'), "w", encoding="utf8") as out_file_ids:
        for i, (query_id, pos_id, neg_id) in tqdm(enumerate(triples)):
            try:
                if collection[pos_id].strip() != "" and collection[neg_id].strip() != "":
                    out_file_ids.write(str(query_id) + "\t" + pos_id + "\t" + neg_id + "\n")
                    out_file_text.write(
                        queries.get(query_id) + "\t" + collection[pos_id] + "\t" + collection[neg_id] + "\n")
            except:
                print('didnt work for {} and {}'.format(pos_id, neg_id))


def read_collection(collection_file_path):
    max_doc_char_length = 1_000_000

    collection = {}
    with open(collection_file_path, "r", encoding="utf8") as collection_file:
        for line in tqdm(collection_file):
            ls = line.split("\t")  # id<\t>text ....
            _id = ls[0]
            max_char_doc = ls[1].rstrip()[:max_doc_char_length]
            collection[_id] = max_char_doc
    return collection

def read_queries(query_file_path):
    queries = {}
    with open(query_file_path, "r", encoding="utf8") as query_file:
        for line in tqdm(query_file):
            ls = line.split("\t")  # id<\t>text ....
            _id = ls[0]
            queries[_id] = ls[1].rstrip()
    return queries

def read_candidates(candidate_path, qrels_path):
    qrels = read_qrels(qrels_path)

    candidates = {}
    with open(candidate_path, "r", encoding="utf8") as candidate_file:

        for line in tqdm(candidate_file):
            ls = line.split()
            if len(ls) == 4:
                query_id = ls[0]
                doc_id = ls[1]
                rank = ls[2]

            else:
                query_id = ls[0]
                doc_id = ls[2]
                rank = ls[3]

            if int(rank) > 100:
                continue
            if doc_id not in collection.keys():
                continue
            if query_id not in qrels.keys():
                continue

            if query_id not in candidates:
                candidates[query_id] = [doc_id]
            else:
                candidates[query_id].append(doc_id)
    return candidates


query_path = "/mnt/c/Users/salthamm/Documents/phd/data/clinical-trials/data/queries_2021.tsv"
collection_path = "/mnt/c/Users/salthamm/Documents/phd/data/clinical-trials/data/collection_all_content.tsv"
candidate_path = "/mnt/c/Users/salthamm/Documents/phd/data/clinical-trials/experiments/wojciech_code/run_output_topics.txt"
qrels_path = "/mnt/c/Users/salthamm/Documents/phd/data/clinical-trials/data/qrels/qrels2021.txt"
output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/clinical-trials/data/train2021_2'

queries = read_queries(query_path)
collection = read_collection(collection_path)

# split the queries in train and test

queries_train_ids = list(queries.keys())[:50]
queries_test_ids = list(set(queries.keys()).difference(set(queries_train_ids)))

print('I have {} train queries and {} test queries'.format(len(queries_train_ids), len(queries_test_ids)))

with open(os.path.join(output_dir, 'queries_train.tsv'), 'w') as out:
    for query_id in queries_train_ids:
        out.write(query_id + '\t' + queries.get(query_id) + '\n')
with open(os.path.join(output_dir, 'queries_test.tsv'), 'w') as out:
    for query_id in queries_test_ids:
        out.write(query_id + '\t' + queries.get(query_id) + '\n')


# first create the training set

create_train_data(queries_train_ids, queries, collection, qrels_path, output_dir)


# read in candidates for test!
candidates = read_candidates(candidate_path, qrels_path)

# queries: query_id \t query_text
# bm25 reranking top100 file: query_id \t doc_id \t query_text \t doc_text
with open(os.path.join(output_dir, 'test_queries_rerank.tsv'), "w", encoding="utf8") as outfile_testrerank:
    for query_id in queries_test_ids:
        query_text = queries.get(query_id)
        candidates_query = candidates.get(query_id)

        for doc_id in candidates_query:
            if doc_id in collection.keys():
                outfile_testrerank.write(query_id + '\t' + doc_id + '\t' +
                                         query_text + '\t' + collection.get(doc_id) + '\n')





