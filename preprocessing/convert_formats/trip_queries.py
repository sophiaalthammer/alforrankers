import os

# out_path = '/newstorage5/salthamm/tripclick/benchmark/queries.train.head.torso.tsv'
# in_path1 = '/newstorage5/salthamm/tripclick/benchmark/queries.head.train.tsv'
# in_path2 = '/newstorage5/salthamm/tripclick/benchmark/queries.torso.train.tsv'
#
# with open(os.path.join(out_path), 'w') as out, open(os.path.join(in_path1),'r') as in1, open(os.path.join(in_path2),'r') as in2:
#     lines1 = in1.readlines()
#     lines2 = in2.readlines()
#
#     for line in lines1:
#         out.write(line)
#     for line in lines2:
#         out.write(line)


# filter the train files from the training file which are not in the queries?

queries = '/newstorage5/salthamm/tripclick/benchmark/queries.train.tsv'
train_file = '/newstorage5/salthamm/tripclick/fix_3bert_ensemble_train_triples.tsv'
train_file_new = '/newstorage5/salthamm/tripclick/fix_3bert_ensemble_train_triples_headtorso.tsv'

with open(train_file_new, 'w') as out_train, open(queries, 'r') as f_queries, open(train_file, 'r') as f_train:
    lines = f_queries.readlines()
    queries = []
    for line in lines:
        queries.append(line.split('\t')[1].rstrip('\n'))
    print('read in queries')

    lines_train = f_train.readlines()
    print('read in lines train')
    for line in lines_train:
        if line.split('\t')[2] in queries:
            out_train.write(line)



