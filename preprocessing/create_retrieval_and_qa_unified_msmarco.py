#
# 
# -------------------------------
#

import random
import argparse
import os
import sys
from fuzzysearch import levenshtein
from tqdm import tqdm
sys.path.append(os.getcwd())
from blingfire import *
from collections import defaultdict
import csv 
import operator
from collections import Counter
import statistics

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from blingfire import *
from matchmaker.core_metrics import load_qrels

# from: https://www.nltk.org/_modules/nltk/tokenize/util.html
def align_tokens(tokens, sentence):
    """
    This module attempt to find the offsets of the tokens in *s*, as a sequence
    of ``(start, end)`` tuples, given the tokens and also the source string.

        >>> from nltk.tokenize import TreebankWordTokenizer
        >>> from nltk.tokenize.util import align_tokens
        >>> s = str("The plane, bound for St Petersburg, crashed in Egypt's "
        ... "Sinai desert just 23 minutes after take-off from Sharm el-Sheikh "
        ... "on Saturday.")
        >>> tokens = TreebankWordTokenizer().tokenize(s)
        >>> expected = [(0, 3), (4, 9), (9, 10), (11, 16), (17, 20), (21, 23),
        ... (24, 34), (34, 35), (36, 43), (44, 46), (47, 52), (52, 54),
        ... (55, 60), (61, 67), (68, 72), (73, 75), (76, 83), (84, 89),
        ... (90, 98), (99, 103), (104, 109), (110, 119), (120, 122),
        ... (123, 131), (131, 132)]
        >>> output = list(align_tokens(tokens, s))
        >>> len(tokens) == len(expected) == len(output)  # Check that length of tokens and tuples are the same.
        True
        >>> expected == list(align_tokens(tokens, s))  # Check that the output is as expected.
        True
        >>> tokens == [s[start:end] for start, end in output]  # Check that the slices of the string corresponds to the tokens.
        True

    :param tokens: The list of strings that are the result of tokenization
    :type tokens: list(str)
    :param sentence: The original string
    :type sentence: str
    :rtype: list(tuple(int,int))
    """
    point = 0
    offsets = []
    for token in tokens:
        try:
            start = sentence.index(token, point)
        except ValueError:
            raise ValueError('substring "{}" not found in "{}"'.format(token, sentence))
        point = start + len(token)
        offsets.append((start, point))
    return offsets

from fuzzysearch import find_near_matches
import concurrent.futures
import json

def generate_pairs():
    for t,(idx,answers) in enumerate(data["answers"].items()):
        if t > 20000:
            break
        for answ in answers:
            if answ != "No Answer Present." and answ != "":
                for passage in data["passages"][idx]:
                    if passage["is_selected"] == 1:
                        yield ((passage["is_selected"],data["query_id"][idx]),answ.lower(),passage["passage_text"].lower())

def generate_qrel_matches(qrels,qa_per_query_id,queries,docs):
    for t,(query,doc) in enumerate(qrels):
        #if t > 20000:
        #    break
        qa_data = qa_per_query_id[query][0]

        query_text = queries[query]
        doc_text = docs[doc]
        if str(query_text.strip()).lower() != str(qa_data[0].strip()).lower():
            print("warning diff query text",query_text,qa_data[0])

        yield ((query,doc),qa_data[1],doc_text.lower())

def hasNumbers(inputString):
    return sum(1 if char.isdigit() else 0 for char in inputString)

def getNumbers(inputString):
    numbers = []
    for char in inputString:
        if char.isdigit(): numbers.append(char)
    return numbers

def match_answer(data):
    meta, answers, passage = data
    result = []
    answer_words=[]

    for answer in answers:
        allowed_dist = len(answer)//4
        nums = getNumbers(answer)
        #    allowed_dist = min(allowed_dist,nums - 1)
        

        fuzzy_match = find_near_matches(answer,passage,max_l_dist=allowed_dist)
        if len(fuzzy_match) > 0:
            for m in fuzzy_match:
                if len(nums) > 0:
                    num_idx = 0
                    for char in m.matched:
                        if char == nums[num_idx]:
                            num_idx+=1
                            if num_idx == len(nums):
                                break
                    if num_idx == len(nums):
                        result.append(m)
                        answer_words.append(len(text_to_words(m.matched).split()))
                        #print("found nums",answer,m.matched)
                    #else:
                        #print("skipped nums",answer,m.matched)
                else:
                    result.append(m)
                    answer_words.append(len(text_to_words(m.matched).split()))
    
    if len(result) > 0:
        return data,True,result,answer_words
    else:
        return data,False,None,None

if __name__ == '__main__':


    #
    # config
    #
    parser = argparse.ArgumentParser()

    parser.add_argument('--query-in', action='store', dest='query_in', required=True)
    parser.add_argument('--docs-in', action='store', dest='docs_in', required=True)
    parser.add_argument('--qrels', action='store',  dest='qrels', required=True)
    
    parser.add_argument('--matched-out', action='store',  dest='out_file', required=True)
    parser.add_argument('--matched-out-text', action='store',  dest='out_file_text', required=True)

    parser.add_argument('--msmarco-qa', action='store', dest='msmarco',
                        help='msmarco_qa json file', required=True)

    args = parser.parse_args()


    #
    # work
    #


    docs = {}
    with open(args.docs_in,"r",encoding="utf8") as in_file:
        for l in tqdm(in_file,desc="Docs"):
            l = l.split("\t")
            docs[l[0]] = l[1].strip()
    
    queries = {}
    with open(args.query_in,"r",encoding="utf8") as in_file:
        for l in tqdm(in_file,desc="Queries"):
            l = l.split("\t")
            queries[l[0]] = l[1].strip()

    qrels = []
    with open(args.qrels,"r",encoding="utf8") as in_file:
        for l in in_file:
            l = l.split()
            qrels.append((l[0],l[2]))

    #def find_all_subarray(main_array, sub_array):
    #    for i in range(len(main_array)-len(sub_array)):
    #        if main_array[i:i+len(sub_array)] == sub_array:
    #            yield i

    #########################
    data = json.load(open(args.msmarco,"r",encoding="utf8"))

    qa_per_query_id = defaultdict(list)

    for t,(idx,query_id) in tqdm(enumerate(data["query_id"].items()),desc="QA"):
        str_q = str(query_id)
        answers = []
        for answ in data["answers"][idx]:
            if answ != "No Answer Present." and answ != "":
                answers.append(answ.lower())
        qa_per_query_id[str_q].append((data["query"][idx],answers))



    msmarco_start_positions_answers = []
    msmarco_answer_lengths = []
    msmarco_skipped_but_present = 0
    msmarco_fuzzy_match=0

    levenshtein_distances = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        with open(args.out_file,"w",encoding="utf8") as out_file,open(args.out_file_text,"w",encoding="utf8") as out_file_text: 
            for data,matched,match,answer_len in tqdm(executor.map(match_answer, generate_qrel_matches(qrels,qa_per_query_id,queries,docs), chunksize=16)):
                if matched:
                    out_file.write(str(data[0][0])+"\t"+str(data[0][1])+"\t"+" ".join([str(m.start)+","+str(m.end) for m in match])+"\n")
                    for m in match:
                        if m.dist > 1: out_file_text.write(str(m.dist)+"\t"+str(data[1])+",["+str(m.matched)+"]\t"+ queries[data[0][0]] +"\t"+ docs[data[0][1]] + "\n")
                    levenshtein_distances.extend([m.dist for m in match])
                    msmarco_start_positions_answers.extend([m.start for m in match])
                    msmarco_answer_lengths.extend(answer_len)
                    msmarco_fuzzy_match += 1
                else:
                    msmarco_skipped_but_present +=1

    total =msmarco_skipped_but_present+msmarco_fuzzy_match
    print("msmarco_fuzzy_match",msmarco_fuzzy_match,"(",msmarco_fuzzy_match/total,")")
    print("msmarco_skipped_but_present",msmarco_skipped_but_present,"(",msmarco_skipped_but_present/total,")")
    print("total",total)

    
    def crappyhist(a, bins=20, width=30,range_=(0,1)):
        h, b = np.histogram(a, bins,range_)

        for i in range (0, bins):
            print('{:12.5f}  | {:{width}s} {}'.format(
                b[i], 
                '#'*int(width*h[i]/np.amax(h)), 
                h[i],#/len(a), 
                width=width))
        print('{:12.5f}  |'.format(b[bins]))

    print("levenshtein_distances (characters)")
    crappyhist(levenshtein_distances,bins=50,range_=(0,50))
    print("msmarco_answer_lengths (words)")
    crappyhist(msmarco_answer_lengths,bins=50,range_=(0,50))
    print("msmarco_start_positions_answers (words)")
    crappyhist(msmarco_start_positions_answers,bins=50,range_=(0,100))

    exit()
    levenshtein_distances = []

    for t,(idx,answers) in tqdm(enumerate(data["answers"].items())):
        if t > 10000:
            break
        for answ in answers:
            if answ != "No Answer Present." and answ != "":

                tokenized_answer = text_to_words(answ.lower()).split()

                for passage in data["passages"][idx]:
                    if passage["is_selected"] == 1:
                        doc_sequence = passage["passage_text"].lower()

                        passage_tokenized = text_to_words(doc_sequence).split()

                        #pos = list(find_all_subarray(passage_tokenized,tokenized_answer))
                        #if len(pos) > 0:
                        #    for p in pos:
                        #        msmarco_answer_lengths.append(len(tokenized_answer))
    #
                        #        p_list = np.arange(0,len(tokenized_answer)) + p
                        #        msmarco_start_positions_answers.extend(list((p_list)/len(passage_tokenized)))
                        #    msmarco_direct_match +=1
                        #    #levenshtein_distances.append(0)
                        #else:
                            # no direct match, lets try fuzzy search
                        fuzzy_match = find_near_matches(answ.lower(),doc_sequence,max_l_dist=len(answ)//4)
                        if len(fuzzy_match) > 0:
                            msmarco_fuzzy_match +=1
                            for m in fuzzy_match:
                                levenshtein_distances.append(m.dist)
                        else:
                            msmarco_skipped_but_present +=1

    total =msmarco_skipped_but_present+msmarco_fuzzy_match

    print("msmarco_fuzzy_match",msmarco_fuzzy_match,"(",msmarco_fuzzy_match/total,")")
    print("msmarco_skipped_but_present",msmarco_skipped_but_present,"(",msmarco_skipped_but_present/total,")")
    print("total",total)

    print("msmarco_answer_lengths",statistics.mean(msmarco_answer_lengths),statistics.stdev(msmarco_answer_lengths))

    def crappyhist(a, bins=20, width=30,range_=(0,1)):
        h, b = np.histogram(a, bins,range_)

        for i in range (0, bins):
            print('{:12.5f}  | {:{width}s} {}'.format(
                b[i], 
                '#'*int(width*h[i]/np.amax(h)), 
                h[i],#/len(a), 
                width=width))
        print('{:12.5f}  |'.format(b[bins]))

    crappyhist(levenshtein_distances,bins=50,range_=(0,50))


    covered_words_per_doc = {}
    annotated_pairs = {}

    all_num_ranges=[]
    all_is_rotated=[]
    all_covered_word_idxs=[]
    all_answer_lengths=[]
    all_relative_coverage=[]

    #for row in list(annotations_tsv)[1:]:
    #    pair = (row[8], row[7])
    #
    #    if row[1] in [label_2,label_3]:
    #        covered_chars,num_ranges = range2annotation(row[2], len(docs[pair[1]][0]))
    #        covered_word_idxs = []
    #        covered_word_idxs_rel = []
    #
    #        for t,offsets in enumerate(docs[pair[1]][2]):
    #            if np.any(covered_chars[offsets[0]:offsets[1]]):
    #                covered_word_idxs.append(t)
    #                covered_word_idxs_rel.append(t/len(docs[pair[1]][1]))
    #
    #        relative_coverage = len(covered_word_idxs) / len(docs[pair[1]][1])
    #        is_rotated = row[3] == "true"
    #        if pair not in covered_words_per_doc:
    #            covered_words_per_doc[pair]=[]
    #        covered_words_per_doc[pair].append((row[1],covered_word_idxs,covered_word_idxs_rel,num_ranges,relative_coverage,is_rotated))
    #
    #        all_num_ranges.append(num_ranges)
    #        all_is_rotated.append(is_rotated)
    #        all_covered_word_idxs.extend(covered_word_idxs)
    #        all_relative_coverage.append(relative_coverage)
    #        all_answer_lengths.append(len(covered_word_idxs))
    #
    #    if pair not in annotated_pairs:
    #        annotated_pairs[pair]=[]
    #    annotated_pairs[pair].append(row)
    #
    #
    #print(Counter(all_num_ranges))
    #print(Counter(all_is_rotated))
    ##print(all_covered_word_idxs)
    #print(statistics.mean(all_relative_coverage),statistics.stdev(all_relative_coverage))
    #print(statistics.mean(all_answer_lengths),statistics.stdev(all_answer_lengths))