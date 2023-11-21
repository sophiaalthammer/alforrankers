#
# get the word representation from a neural-ir model
# -------------------------------


import argparse
import copy
import os
import gc
import glob
import time
import sys
sys.path.append(os.getcwd())

# needs to be before torch import 
from allennlp.common import Tqdm
import json
import numpy
from blingfire import *

from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy import stats
from matplotlib.ticker import FormatStrFormatter,PercentFormatter

Tqdm.default_mininterval = 1

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams.update({'font.size': 15})
plt.rc('legend',**{'fontsize':13})
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['svg.fonttype'] = "none" 
mpl.rcParams['grid.linewidth'] = 1.5
mpl.rcParams['lines.linewidth'] = 2

# config
# -------------------------------
#
# trained model folder, new experiment folder, own config? 
parser = argparse.ArgumentParser()

parser.add_argument('--msmarco', action='store', dest='msmarco',
                    help='msmarco_qa json file', required=True)

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='the out file', required=True)


args = parser.parse_args()


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += 1# len(sub) # use start += 1 to find overlapping matches

def find_all_subarray(main_array, sub_array):
    for i in range(len(main_array)-len(sub_array)):
        if main_array[i:i+len(sub_array)] == sub_array:
            yield i

#
# msmarco
#

data = json.load(open(args.msmarco,"r"))


msmarco_start_positions_answers_ids = {}
msmarco_start_positions_answers_absolute_ids = {}

msmarco_start_positions_answers = []
msmarco_start_positions_answers_absolute = []
msmarco_skipped_but_present = 0
msmarco_used=0


relative_binned_ids = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}

for idx,answers in Tqdm.tqdm(data["answers"].items()):
    for answ in answers:
        if answ != "No Answer Present." and answ != "":

            tokenized_answer = text_to_words(answ).split()

            for passage in data["passages"][idx]:
                if passage["is_selected"] == 1:# and len(passage["passage_text"]) > 500:

                    passage_tokenized = text_to_words(passage["passage_text"]).split()

                    #pos = passage["passage_text"].find(answ)
                    #pos = list(find_all(passage["passage_text"][:10_000],answ))
                    pos = list(find_all_subarray(passage_tokenized,tokenized_answer))
                    if len(pos) > 0:
                        #for p in pos:
                        p = pos[-1]
                        msmarco_start_positions_answers.append((p + len(tokenized_answer)/2)/len(passage_tokenized))
                        msmarco_start_positions_answers_absolute.append(p + len(tokenized_answer)/2)

                        if idx not in msmarco_start_positions_answers_ids:
                            msmarco_start_positions_answers_ids[idx] = []
                            msmarco_start_positions_answers_absolute_ids[idx] = []
                        msmarco_start_positions_answers_ids[idx].append((p + len(tokenized_answer)/2)/len(passage_tokenized))
                        msmarco_start_positions_answers_absolute_ids[idx].append(p + len(tokenized_answer)/2)
                        relative_binned_ids[int((p + len(tokenized_answer)/2)/len(passage_tokenized)*10)].append(idx)

                        msmarco_used +=1
                    else:
                        msmarco_skipped_but_present +=1



def crappyhist(a, bins=10,range_=(0,1), width=100):
    h, b = numpy.histogram(a, bins,range=range_)

    for i in range (0, bins):
        print('{:12.5f}  | {:{width}s} {}'.format(
            b[i], 
            '#'*int(width*h[i]/numpy.amax(h)), 
            h[i],#/len(a), 
            width=width))
    print('{:12.5f}  |'.format(b[bins]))

min_bin_len = 2000
for i in range(10):
    min_bin_len = min(min_bin_len,len(relative_binned_ids[i]))

with open(args.out_file,"w",encoding="utf8") as out_file:

    import random
    for i in range(5,10):
        sampled_ids = random.sample(range(len(relative_binned_ids[i])), k=min_bin_len)
        for s in sampled_ids:
            qid = relative_binned_ids[i][s]
            out_file.write(str(data["query_id"][qid])+"\t"+str(data["query"][qid])+"\n")

