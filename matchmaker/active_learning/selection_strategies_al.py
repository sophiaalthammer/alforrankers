import os
import random
#from pyserini.search import LuceneSearcher
from matchmaker.eval import evaluate_model
from matchmaker.active_learning.generate_training_subset import load_file
from matchmaker.generation_pipe import execute_and_return_run_folder

from itertools import combinations

import os
import numpy as np
import copy
import time
import glob
from typing import Dict, Tuple, List
from rich.console import Console
from rich.live import Live

console = Console()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy

from allennlp.nn.util import move_to_device
from allennlp.common import Params, Tqdm

from matchmaker.utils.core_metrics import *

from matchmaker.utils.cross_experiment_cache import *
from matchmaker.utils.input_pipeline import *
from matchmaker.utils.performance_monitor import *
import time
from matchmaker.active_learning.utils import *

def qbc_selection(committee_folders, current_train_size, run_folder, iteration, config, train_file_path_subset,
                  train_file_original):
    print('start with comittee rankings')
    # get committee rankings
    commitee_rankings = get_committee_rankings(committee_folders)

    # compute vote entropy
    query_ids_ordered_ve, vote_entropies = compute_vote_entropy(commitee_rankings)

    print('finished with vote entropies')
    with open(os.path.join(run_folder, 'vote_entropies_iteration{}.txt'.format(iteration)), 'w') as out_file_ve:
        for ve in vote_entropies:
            out_file_ve.write(str(ve[0]) + '\t' + str(ve[1]) + '\n')

    # filter out the queries in the train file
    query_ids_ordered_ve = filter_queries_from_train(query_ids_ordered_ve, config)
    print('filtered out the queries which are already in train')

    add_queries_by_ids(query_ids_ordered_ve, config, train_file_path_subset, train_file_original, run_folder, iteration,
                       selection_strategy=config['al_selection_annotation'])

    current_train_size += int(config['no_queries_added_iteration'])
    return current_train_size

def uncertainty_selection(run_folder_iteration, current_train_size, config, train_file_path_subset, train_file_original,
                          run_folder, iteration):
    print('start with measuring the uncertainty')
    # get the rankings with the score
    rankings = get_ranking_scores(run_folder_iteration, 'test_train_top100_bm25-output.txt')

    # filter out the q_d pairs i already have in the train triple
    rankings = filter_q_d_train(rankings, config)
    print('now i am finished with filtering')

    # now select the query document pairs which are most uncertain
    rankings_format = {}
    for q_id, dict_docs in rankings.items():
        for doc_id, score in dict_docs.items():
            rankings_format.update({(q_id, doc_id): score})

    q_d_pairs_selected, no_added_queries_2 = uncertainty_select_q_d_pairs(rankings_format, config)

    # get the train triple
    queries_train_mapping = get_query_mapping(config['query_tsv'])
    if 'tripclick' in train_file_original:
        # reading in with cut off of top 200
        doc_train_mapping = get_query_mapping(config['collection_tsv'], cut=True)
        doc_train_mapping_reverse = get_query_mapping(config['collection_tsv'], reverse=True, cut=True)
    else:
        doc_train_mapping = get_query_mapping(config['collection_tsv'])
        doc_train_mapping_reverse = get_query_mapping(config['collection_tsv'], reverse=True)
    reranked_results = read_reranking_file(config['test']['train_top100_bm25']['tsv'])

    qrels = read_qrels(config['qrels_train'])

    no_queries_added = add_query_doc_ids(config, q_d_pairs_selected, reranked_results, qrels,
                                         queries_train_mapping, doc_train_mapping, doc_train_mapping_reverse,
                                         train_file_path_subset, train_file_original, run_folder, iteration)

    print('added {} to the training file'.format(no_queries_added))
    current_train_size += int(no_queries_added)
    print('this is the current train size {}'.format(current_train_size))
    return current_train_size, no_added_queries_2


def diversity_selection(run_folder_iteration, current_train_size, config, train_file_path_subset, train_file_original,
                        run_folder, iteration):
    # do the dense retrieval embedding and then cluster the
    print('start encoding and clustering the queries with the new trained model')
    config_overwrites_text_dr = "trained_model: {}, expirement_base_path: {}, faiss_index_type: ivf, " \
                                "faiss_ivf_list_count: {}".format(run_folder_iteration, run_folder_iteration,
                                                         int(config['no_queries_added_iteration']) * 4)
    index_dir = execute_and_return_run_folder(["python", "matchmaker/clustering.py", "encode+index+search", "--config",
                                               config["clustering_config"], "--run-name", "queries_clustering",
                                               "--config-overwrites", config_overwrites_text_dr])

    if not index_dir.startswith('/') and '/' in index_dir:
        index_dir = index_dir[index_dir.find('/'):]
    # read the clusters
    print('finished with clustering now read in clusters')
    with open(os.path.join(index_dir, 'clusters.p'), 'rb') as f:
        clusters = pickle.load(f)

    # create mapping in order to get real ids in clusters
    seq_ids, id_mapping, doc_infos, storage_filled_to_index = load_seq_ids(index_dir)
    mapping = create_id_mapping(id_mapping, seq_ids, storage_filled_to_index)
    clusters_real_ids = get_cluster_real_ids(clusters, mapping)
    clusters_real_ids = filter_clusters_from_train(clusters_real_ids, config)

    print('number of clusters is now {}'.format(len(clusters_real_ids)))

    assert len(clusters_real_ids) == int(config['no_queries_added_iteration']) * 4

    query_ids_added = [random.sample(one_cluster_ids, 1)[0] for one_cluster_ids in clusters_real_ids if
                       len(one_cluster_ids) > 0]
    print('number of query ids from clusters is {}'.format(len(query_ids_added)))
    add_queries_by_ids(query_ids_added, config, train_file_path_subset, train_file_original, run_folder, iteration,
                       selection_strategy=config['al_selection_annotation'])

    current_train_size += int(config['no_queries_added_iteration'])
    return current_train_size

def give_position_in_list(ranking, doc_id):
    try:
        position = ranking.index(doc_id)
    except:
        position = 2000
    return position


def compute_vote_entropy(commitee_rankings):
    # only have the same queries in the ranking
    first_member = list(commitee_rankings.keys())[0]
    queries_first_member = commitee_rankings.get(first_member).keys()
    for member_name, rankings in commitee_rankings.items():
        queries_first_member = set(queries_first_member).intersection(set(rankings.keys()))

    # check if the query_text is the same
    vote_entropies = []
    j = 0
    for query_id in list(queries_first_member):
        # only have the same documents in the ranking
        doc_list = []
        for comitee_member, rankings in commitee_rankings.items():
            doc_list.extend(rankings.get(query_id))
        doc_list = list(set(doc_list))

        # compute vote entropy
        ve_pairs = []
        for pair in list(combinations(doc_list, 2)):
            partial_order_count = 0
            partial_order_reverse_count = 0
            for comitee_member, rankings in commitee_rankings.items():
                commitee_ranking = rankings.get(query_id)

                # first do the order of the pair
                if give_position_in_list(commitee_ranking, pair[0]) >= give_position_in_list(commitee_ranking,
                                                                                             pair[1]):
                    partial_order_count += 1
                else:
                    partial_order_reverse_count += 1
                # then do the order of the other pair
            ve_pairs.extend([partial_order_count, partial_order_reverse_count])


        vote_entropy_query = -1 * len(commitee_rankings.keys()) * sum(np.multiply([ve for ve in ve_pairs if ve !=0],np.log(
            [ve/len(commitee_rankings.keys()) for ve in ve_pairs if ve != 0])))
        vote_entropies.append((query_id, vote_entropy_query))


    vote_entropies.sort(key=lambda x: x[1], reverse=True)
    query_ids_ordered_ve = [x[0] for x in vote_entropies]
    return query_ids_ordered_ve, vote_entropies


def filter_queries_from_train(query_ids_ordered_ve, config):
    # get mapping query_text to query_id
    queries_train = {}
    with open(config['query_tsv'], 'r') as file:
        lines = file.readlines()
        for line in lines:
            splitted = line.split('\t')
            queries_train.update({splitted[1].rstrip('\n'): splitted[0]})

    # queries already in training file
    query_text_in_train = []
    with open(config['train_tsv'], 'r') as train_file:
        lines = train_file.readlines()
        for line in lines:
            query_text = line.split('\t')[0]
            query_text_in_train.append(query_text)

    no_removed = 0
    print('length of queries is now {}'.format(len(query_ids_ordered_ve)))
    #for query_text in query_text_in_train:
    #    if queries_train.get(query_text) and queries_train.get(query_text) in query_ids_ordered_ve:
    #        query_ids_ordered_ve.remove(queries_train.get(query_text))
    #        no_removed += 1
    query_ids_ordered_ve = [id for id in query_ids_ordered_ve if not id in [queries_train.get(query_text)
                                                                            for query_text in query_text_in_train
                                                                            if queries_train.get(query_text)]]
    print('removed {} query ids from the list since they are already in train set'.format(no_removed))
    print('length of queries is now {} after filtering train queries'.format(len(query_ids_ordered_ve)))
    return query_ids_ordered_ve


def filter_q_d_train(rankings, config):
    # get mapping query_text to query_id
    queries_train = {}
    with open(config['query_tsv'], 'r') as file:
        lines = file.readlines()
        for line in lines:
            splitted = line.split('\t')
            queries_train.update({splitted[1].rstrip('\n'): splitted[0]})

    # queries already in training file
    query_text_in_train = []
    with open(config['train_tsv'], 'r') as train_file:
        lines = train_file.readlines()
        for line in lines:
            query_text = line.split('\t')[0]
            query_text_in_train.append(query_text)

    no_removed = 0
    print('length of queries is now {}'.format(len(rankings)))

    list_query_ids_train = [queries_train.get(query_text) for query_text in query_text_in_train
                            if queries_train.get(query_text)]
    print('there are {} query ids in train'.format(len(list_query_ids_train)))

    for q_id in list_query_ids_train:
        if rankings.get(q_id):
            rankings.pop(q_id)
    print('finished with filtering')
    print('length of rankings is now {}'.format(len(rankings.keys())))
    return rankings


def filter_clusters_from_train(clusters, config):
    # get mapping query_text to query_id
    queries_train = {}
    with open(config['query_tsv'], 'r') as file:
        lines = file.readlines()
        for line in lines:
            splitted = line.split('\t')
            queries_train.update({splitted[1].rstrip('\n'): splitted[0]})

    # queries already in training file
    query_text_in_train = []
    with open(config['train_tsv'], 'r') as train_file:
        lines = train_file.readlines()
        for line in lines:
            query_text = line.split('\t')[0]
            query_text_in_train.append(query_text)

    print('now i have in total {} ids in the clusters'.format(len([item for sublist in clusters for item in sublist])))
    train_query_ids = [queries_train.get(query_text) for query_text in query_text_in_train
                       if queries_train.get(query_text)]
    clusters = [[id for id in cluster if id not in train_query_ids] for cluster in clusters]
    print('now i have in total {} ids in the clusters'.format(len([item for sublist in clusters for item in sublist])))
    return clusters


def get_committee_rankings(committee_folders):
    commitee_rankings = {}
    for commitee_member in committee_folders:
        rankings = get_ranking(commitee_member, 'test_train_top100_bm25-output.txt')
        member_name = commitee_member.split('/')[-1]
        commitee_rankings.update({member_name: rankings})
    return commitee_rankings

def get_ranking(run_folder, file_name, score=False):
    rankings = {}
    with open(os.path.join(run_folder, file_name), 'r') as f:
        # read them in q_id \t doc_id \t rank \t score
        flines = f.readlines()
        for line in flines:
            line = line.split('\t')

            q_id = line[0]
            doc_id = line[1]
            # rank = line[2]
            score_item = float(line[3].rstrip('\n'))
            if not rankings.get(q_id):
                rankings.update({q_id: []})

            doc_list = rankings.get(q_id)

            if score:
                doc_list.append((doc_id, score_item))
                rankings.update({q_id: doc_list})
            else:
                doc_list.append(doc_id)
                rankings.update({q_id: doc_list})
    return rankings


def get_ranking_scores(run_folder, file_name):
    rankings = {}
    with open(os.path.join(run_folder, file_name), 'r') as f:
        # read them in q_id \t doc_id \t rank \t score
        flines = f.readlines()
        for line in flines:
            line = line.split('\t')

            q_id = line[0]
            doc_id = line[1]
            # rank = line[2]
            score_item = float(line[3].rstrip('\n'))
            if not rankings.get(q_id):
                rankings.update({q_id:{}})
            rankings.get(q_id).update({doc_id: score_item})
    return rankings


def findClosest_position(arr, n, target, mapping_scores):
    # code from https://www.geeksforgeeks.org/find-closest-number-array/
    # Corner cases
    if (target <= mapping_scores.get(arr[0])):
        return 0
    if (target >= mapping_scores.get(arr[n - 1])):
        return n - 1

    # Doing binary search
    i = 0
    j = n
    mid = 0
    while (i < j):
        mid = (i + j) // 2

        if (mapping_scores.get(arr[mid]) == target):
            return mid

        # If target is less than array
        # element, then search in left
        if (target < mapping_scores.get(arr[mid])):

            # If target is greater than previous
            # to mid, return closest of two
            if (mid > 0 and target > mapping_scores.get(arr[mid - 1])):
                return getClosest(mid - 1, mid, arr, target, mapping_scores)

            # Repeat for left half
            j = mid

        # If target is greater than mid
        else:
            if (mid < n - 1 and target < mapping_scores.get(arr[mid + 1])):
                return getClosest(mid, mid + 1, arr, target, mapping_scores)
            # update i
            i = mid + 1

    # Only single element left after search
    return mid


def getClosest(val1, val2, arr, target, mapping_scores):
    # code from https://www.geeksforgeeks.org/find-closest-number-array/
    if (target - mapping_scores.get(arr[val1]) >= mapping_scores.get(arr[val2]) - target):
        return val2
    else:
        return val1

def get_query_mapping(path, reverse=False, cut=False):
    # mapping query_id to text
    queries_train_mapping = {}
    with open(path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            splitted = line.split('\t')
            if not reverse:
                text = splitted[1].rstrip('\n')
                if cut:
                    text = ' '.join(text.split(' ')[:200])
                queries_train_mapping.update({splitted[0]: text})
            else:
                text = splitted[1].rstrip('\n')
                if cut:
                    text = ' '.join(text.split(' ')[:200])
                queries_train_mapping.update({text: splitted[0]})
    return queries_train_mapping


def read_qrels(qrels_file_path):
    qrels = {}
    with open(qrels_file_path, 'r') as qrels_file:
        lines = qrels_file.readlines()

        for line in lines:
            line = line.split('\t')
            query_id = line[0]
            doc_id = line[2]
            rel_grade = int(line[3])
            if qrels.get(query_id):
                if qrels.get(query_id).get(rel_grade):
                    qrels.get(query_id).get(rel_grade).append(doc_id)
                else:
                    qrels.get(query_id).update({rel_grade:[doc_id]})
            else:
                qrels.update({query_id: {rel_grade:[doc_id]}})

    return qrels


def load_seq_ids(encode_folder):
    dfs = np.load(os.path.join(encode_folder, "doc_infos.npz"), allow_pickle=True)
    doc_infos = dfs.get("doc_infos")[()]
    id_mapping = dfs.get("id_mapping")[()]
    seq_ids = dfs.get("seq_ids")[()]
    storage_filled_to_index = dfs.get("storage_filled_to_index")[()]
    return seq_ids, id_mapping, doc_infos, storage_filled_to_index

def create_id_mapping(id_mapping, seq_ids, storage_filled_to_index):
    # create id mapping dictionary
    mapping = {}
    len_shard_total = 0
    for shard_no in range(len(id_mapping)):
        shard = list(id_mapping[shard_no])
        for list_no in range(len(shard)):
            mapping.update({shard[list_no]:seq_ids[len_shard_total+list_no]})
        len_shard_total += storage_filled_to_index[shard_no]
    return mapping

def get_cluster_real_ids(clusters, mapping):
    #replace ids with actual ids for analysis!
    clusters_real_ids = []
    for clust in clusters:
        cluster_with_ids = []
        for id in clust:
            cluster_with_ids.append(mapping.get(id))
        clusters_real_ids.append(cluster_with_ids)
    return clusters_real_ids

def uncertainty_select_q_d_pairs(rankings, config):
    # get the mean and then sample q-d pairs with the mean score!
    scores = [score for q_id, score in rankings.items()]

    print('this is the length of the scores {}'.format(len(scores)))
    scores.sort()

    rankings_sorted = dict(sorted(rankings.items(), key=lambda item: item[1]))
    sorted_q_d = [key for key, value in rankings_sorted.items()]

    position_middle_scores = findClosest_position(sorted_q_d, len(scores), np.mean(scores), rankings_sorted)

    assert position_middle_scores <= len(sorted_q_d)

    q_d_pairs_selected = [sorted_q_d[position_middle_scores]]
    no_added_queries = 1
    for i in range(math.ceil(len(sorted_q_d))):
        if no_added_queries >= int(config['no_queries_added_iteration']) * 4:
            break

        if position_middle_scores - i >= 0:
            q_d_pairs_selected.append(sorted_q_d[position_middle_scores - i])
            no_added_queries += 1

        if no_added_queries >= int(config['no_queries_added_iteration']) * 4:
            break

        if position_middle_scores + i <= len(sorted_q_d):
            q_d_pairs_selected.append(sorted_q_d[position_middle_scores + i])
            no_added_queries += 1

    print('i have select {} query-document pairs which are the most uncertain'.format(len(q_d_pairs_selected)))
    return q_d_pairs_selected, no_added_queries


def read_reranking_file(reranking_path, text=False):
    reranking_file = {}
    with open(reranking_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            line = line.split('\t')
            query_id = line[0]
            doc_id = line[1]
            doc_text = line[3].rstrip('\n')

            if not reranking_file.get(query_id):
                reranking_file.update({query_id: []})
            if text:
                reranking_file[query_id].append(doc_text)
            else:
                reranking_file[query_id].append(doc_id)
    return reranking_file

def get_rank_from_text(doc_train_mapping, rel_text, reranked_results, query_id):
    if doc_train_mapping.get(rel_text):
        pos_doc_id = doc_train_mapping.get(rel_text)
        if pos_doc_id in reranked_results.get(query_id):
            rank_pos_doc_id = reranked_results.get(query_id).index(pos_doc_id)
        else:
            rank_pos_doc_id = 200 # 200 so i know that the text is not in topn, but not higher since i dont want to
    else:
        rank_pos_doc_id = 300
    return rank_pos_doc_id


def add_queries_by_ids(query_ids_added, config, train_file_path_subset, train_file_original, run_folder, iteration,
                       selection_strategy='random'):
    queries_train_mapping = get_query_mapping(config['query_tsv'])

    # read in the reranking file, get dict with query_ids and doc_text! in list
    reranked_results = read_reranking_file(config['test']['train_top100_bm25']['tsv'])
    doc_train_mapping = get_query_mapping(config['collection_tsv'], reverse=True)

    no_queries_added = 0
    # we need train and test to overlap
    print('this is the train file')  # maybe this is the wrong one here...
    with open(train_file_path_subset, 'a') as out_file, open(train_file_original, 'r') as in_file, \
            open(os.path.join(run_folder, 'added_queries_iteration{}.tsv'.format(iteration)),'w') as queries_added_file, \
            open(os.path.join(run_folder, 'added_queries_annotation_effort{}.tsv'.format(iteration)), 'w') as annotation_file:
        # read in train file
        train_lines = in_file.readlines()
        train_triples = {}
        for line in train_lines:
            line = line.split('\t')
            if len(line) == 5:
                if not train_triples.get(line[2]):
                    train_triples.update({line[2]: []})
                train_triples.get(line[2]).append([line[3], line[4]])
            else:
                if not train_triples.get(line[0]):
                    train_triples.update({line[0]: []})
                train_triples.get(line[0]).append([line[1], line[2]])

        for query_id in query_ids_added:
            if no_queries_added >= int(config['no_queries_added_iteration']):
                break

            if queries_train_mapping.get(query_id) and reranked_results.get(query_id):
                query_text = queries_train_mapping.get(query_id)

                if train_triples.get(query_text):
                    if selection_strategy == 'first':
                        # takes the triple where the positive document appears as first in the ranked list
                        if len(train_triples.get(query_text)) > 1:
                            try:
                                rankings_docs = {}
                                for tuple in train_triples.get(query_text):
                                    rank_pos_doc_id = get_rank_from_text(doc_train_mapping, tuple[0],
                                                                         reranked_results, query_id)
                                    rankings_docs.update({rank_pos_doc_id: tuple})
                                min_rank = min(rankings_docs.keys())
                                triple = rankings_docs.get(min_rank)
                            except:
                                triple = random.sample(train_triples.get(query_text), 1)[0]
                        else:
                            triple = random.sample(train_triples.get(query_text), 1)[0]
                    elif selection_strategy == 'random':
                        triple = random.sample(train_triples.get(query_text), 1)[0]

                    # compute annotation effort to get this triple, simulate annotation from top down until you find
                    # the positive one
                    # we assume negatives are for free annotation since they come from bm25, or are generated in the
                    # annotation process anyways

                    rank_pos_doc_id = get_rank_from_text(doc_train_mapping, triple[0], reranked_results, query_id)
                    rank_neg_doc_id = get_rank_from_text(doc_train_mapping, triple[1], reranked_results, query_id)
                    # here also control how the negative is chosen: maybe also takes the first negative one it finds?

                    out_file.write(str(query_text) + '\t' + str(triple[0]) + '\t' + str(triple[1].rstrip('\n')) + '\n')
                    queries_added_file.write(query_id + '\n')
                    annotation_file.write(str(rank_pos_doc_id) + '\t' + str(rank_neg_doc_id) + '\n')
                    no_queries_added += 1

    print('added {} to the training file'.format(no_queries_added))


def add_query_doc_ids(config, q_d_pairs_selected, reranked_results, qrels, queries_train_mapping, doc_train_mapping,
                      doc_train_mapping_reverse, train_file_path_subset, train_file_original, run_folder, iteration):
    no_queries_added = 0
    with open(train_file_path_subset, 'a') as out_file, open(train_file_original, 'r') as in_file, \
            open(os.path.join(run_folder, 'added_queries_iteration{}.tsv'.format(iteration)),
                 'w') as queries_added_file, open(
        os.path.join(run_folder, 'added_queries_annotation_effort{}.tsv'.format(iteration)), 'w') \
            as annotation_file:
        # read in train file
        train_lines = in_file.readlines()
        train_triples = {}
        for line in train_lines:
            line = line.split('\t')
            if len(line) == 5:
                index = 2
            else:
                index = 0
            print('this is the index {}, should be 0 for marco and 2 for trip'.format(index))
            pos_text = line[1 + index]
            neg_text = line[2 + index].rstrip('\n')
            if index == 2:
                pos_text = ' '.join(pos_text.split(' ')[:200])
                neg_text = ' '.join(neg_text.split(' ')[:200])
            if not train_triples.get((line[0+index])):
                train_triples.update({line[0+index]: {pos_text: [neg_text]}})
            else:
                if train_triples.get(line[0+index]).get(pos_text):
                    train_triples.get(line[0+index]).get(pos_text).append(neg_text)
                else:
                    train_triples.get(line[0+index]).update({pos_text: [neg_text]})

        for (query_id, doc_id) in q_d_pairs_selected:
            print('this is the query id {} and the doc id {}'.format(query_id, doc_id))
            if reranked_results.get(query_id):
                print('i got the reranked result for this query id')
                if no_queries_added >= int(config['no_queries_added_iteration']):
                    break

                # check if doc_id is positive or negative
                if not qrels.get(query_id):
                    print('query_id {} is not in qrels'.format(query_id))
                else:
                    rel_grade_doc = None
                    for rel_grade, doc_list in qrels.get(query_id).items():
                        if doc_id in doc_list:
                            rel_grade_doc = rel_grade
                    if rel_grade_doc == None:
                        rel_grade_doc = 0

                    # now here the two cases of the doc_id being pos or negative!
                    if queries_train_mapping.get(query_id) and doc_train_mapping.get(doc_id):
                        query_text = queries_train_mapping.get(query_id)

                        if train_triples.get(query_text):
                            if rel_grade_doc > 0:
                                if train_triples.get(query_text).get(doc_train_mapping.get(doc_id)):
                                    neg_text = train_triples.get(query_text).get(doc_train_mapping.get(doc_id))[0]
                                    if doc_train_mapping.get(doc_id):
                                        rank_neg_doc_id = get_rank_from_text(doc_train_mapping_reverse,
                                                                             str(neg_text), reranked_results, query_id)
                                    else:
                                        rank_neg_doc_id = 200

                                    out_file.write(str(query_text) + '\t' + str(doc_train_mapping.get(doc_id)) + '\t'
                                                   + str(neg_text) + '\n')
                                    queries_added_file.write(query_id + '\n')
                                    no_queries_added += 1
                                    annotation_file.write(str(0) + '\t' + str(rank_neg_doc_id) + '\n')
                                else:
                                    print('didnt find the train triple with the text')
                            else:
                                if len(train_triples.get(query_text).keys()) >= 1:
                                    if config['al_selection_annotation'] == 'first':
                                        # takes the triple where the positive document appears as first in the ranked list
                                        if len(train_triples.get(query_text).keys()) > 1:
                                            try:
                                                rankings_docs = {}
                                                for rel_text in train_triples.get(query_text).keys():
                                                    rank_pos_doc_id = get_rank_from_text(doc_train_mapping_reverse,
                                                                                         rel_text, reranked_results,
                                                                                         query_id)
                                                    rankings_docs.update({rank_pos_doc_id: rel_text})
                                                min_rank = min(rankings_docs.keys())
                                                rel_doc_text = rankings_docs.get(min_rank)
                                            except:
                                                rel_doc_text = random.sample(train_triples.get(query_text).keys(), 1)[0]
                                        else:
                                            rel_doc_text = random.sample(train_triples.get(query_text).keys(), 1)[0]
                                    elif config['al_selection_annotation'] == 'random':
                                        rel_doc_text = random.sample(train_triples.get(query_text).keys(), 1)[0]

                                    rank_pos_doc_id = get_rank_from_text(doc_train_mapping_reverse, rel_doc_text,
                                                                         reranked_results, query_id)
                                    rank_neg_doc_id = get_rank_from_text(doc_train_mapping_reverse,
                                                                         str(doc_train_mapping.get(doc_id)),
                                                                         reranked_results, query_id)

                                    out_file.write(str(query_text) + '\t' + str(rel_doc_text) + '\t' +
                                                   str(doc_train_mapping.get(doc_id)) + '\n')
                                    queries_added_file.write(query_id + '\n')
                                    no_queries_added += 1
                                    annotation_file.write(str(rank_pos_doc_id) + '\t' + str(rank_neg_doc_id) + '\n')
                        else:
                            print('couldnt find a train triple for {}'.format(query_text))

    return no_queries_added


