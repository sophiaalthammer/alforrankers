import argparse
from tqdm import tqdm
import os

#
# config
#
parser = argparse.ArgumentParser()
parser.add_argument('--in-dir', action='store', dest='in_dir',
                    help='input directory', required=False,default="/mnt/c/Users/salthamm/Documents/phd/data/out-of-domain/retrieval_experiments")
args = parser.parse_args()

#
# work
#
in_dir = args.in_dir
#in_dir = "/mnt/c/Users/salthamm/Documents/phd/data/beir"

dirs = os.listdir(in_dir)
all_scores = {}

with open(os.path.join(in_dir, 'collection_all.tsv'), 'w') as out_file:
    for dir in dirs:
        if os.path.isdir(os.path.join(in_dir, dir)):
            if os.path.exists(os.path.join(in_dir, dir, 'collection.tsv')):
                with open(os.path.join(in_dir, dir, 'collection.tsv'), 'r') as f:
                    lines = f.readlines()

                    for l in lines:
                        l = l.split("\t")

                        if len(l) == 2:
                            line = l[0] +"\t" + l[1]
                            out_file.write(line)
                print('Finished with collection file {}'.format(os.path.join(in_dir, dir, 'collection.tsv')))





