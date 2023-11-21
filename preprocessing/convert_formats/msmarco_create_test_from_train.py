import os
import random
#from matchmaker.utils.generate_training_subset import load_qrels, load_file
random.seed(19783624783726)

def load_file(path):
    collection = {}
    with open(path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            splitted = line.split('\t')
            collection.update({splitted[0]: splitted[1].rstrip('\n')})
    return collection

# stop i need to make sure that they are in the qrels!
path = "/newstorage5/salthamm/msmarco/data" #"/mnt/c/Users/salthamm/Documents/phd/data/msmarco-passage/data"

# # open the qrels
# qrels = load_qrels(os.path.join(path, 'qrels.train.tsv'))
# # loads only relevant docs
#
# # load the train queries
# queries_train = load_file(os.path.join(path, 'queries.train.tsv'))
#
# # maybe first select the train queries which are in the qrels, yes thats a good idea!
# queries_train_qrels = list(set(queries_train.keys()).intersection(set(qrels.keys())))
#
# with open(os.path.join(path, 'queries.train.tsv'), 'r') as f:
#     lines = f.readlines()
#
#     lines_test = random.sample(queries_train_qrels, 7000)
#     lines_train = list(set(queries_train_qrels) - set(lines_test))
#
#     with open(os.path.join(path, "data_efficiency/queries.train.wotest.tsv"), "w") as out1:
#         for id in lines_train:
#             out1.write(id + '\t' + queries_train.get(id) + '\n')
#     with open(os.path.join(path, "data_efficiency/queries.test.tsv"), "w") as out1:
#         for id in lines_test:
#             out1.write(id + '\t' + queries_train.get(id) + '\n')


# create test reranking dataset from the test bm25 top100
# i need: query_id, doc_id, query_sequence, doc_sequence = line_parts
# this it the reranking setup, splitted by \t

# load the collection
collection = load_file(os.path.join(path, 'collection.tsv'))

queries = load_file(os.path.join(path, 'data_efficiency/queries.test.tsv'))

# load the bm25 ret file
with open(os.path.join(path, 'data_efficiency/queries.test.bm25.reranking2.tsv'), 'w') as out_file, \
    open(os.path.join(path, 'data_efficiency/queries.test.bm25_top100.txt'), 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split(' ')
        query_id = line[0]
        doc_id = line[2]

        query_text = queries.get(query_id)
        doc_text = collection.get(doc_id)

        out_file.write(query_id +'\t'+ doc_id +'\t'+ query_text +'\t'+ doc_text + '\n')


#
# # ms marco load training triples, then exclude the triples from the test set
#
# test_queries_path = os.path.join(path, "data_efficiency/queries.test.tsv")
# test_queries = load_file(test_queries_path)
#
# test_queries_text = [text.rstrip('\n') for text in list(test_queries.values())]
#
# triples_path = "/newstorage5/salthamm/msmarco/data/train/triples.train.small-split4"
#
# # load the triples
# counter_in = 0
# counter_out = 0
# with open(os.path.join(triples_path, 'joined.wotest.tsv'), 'w') as out_file, open(os.path.join(triples_path, 'joined.tsv'), 'r') as f:
#     lines = f.readlines()
#
#     for line in lines:
#         query = line.split('\t')[0]
#         if query not in test_queries_text:
#             out_file.write(line)
#             counter_in += 1
#         else:
#             counter_out += 1
#
#     print('finished with {} in the triples and {} left out because they are in the test file'.format(counter_in, counter_out))
#












