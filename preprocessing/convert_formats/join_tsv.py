import os

path = '/newstorage5/salthamm/msmarco/data/dev.subset.bm25_plain_top100-split6'

files = os.listdir(path)
with open(os.path.join(path, 'joined.tsv'), 'w') as out_file:
    for file in files:
        with open(os.path.join(path, file), 'r') as in_file1:
            lines = in_file1.readlines()

            for line in lines:
                out_file.write(line)



