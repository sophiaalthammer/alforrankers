import os

qrels_path = '/mnt/c/Users/salthamm/Documents/phd/data/tripclick/data/qrels'

qrels_lines = []
with open(os.path.join(qrels_path, 'qrels.dctr.head.train.txt'), 'r') as f:
    lines = f.readlines()

    for line in lines:
        line = line.split(' ')
        qrels_lines.append(line)

with open(os.path.join(qrels_path, 'qrels.raw.torso.train.txt'), 'r') as f:
    lines = f.readlines()

    for line in lines:
        line = line.split(' ')
        qrels_lines.append(line)

with open(os.path.join(qrels_path, 'qrels.raw.tail.train.txt'), 'r') as f:
    lines = f.readlines()

    for line in lines:
        line = line.split(' ')
        qrels_lines.append(line)

with open(os.path.join(qrels_path, 'qrels.train.tsv'), 'w') as out:
    for line in qrels_lines:
        out.write(line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + line[3])
