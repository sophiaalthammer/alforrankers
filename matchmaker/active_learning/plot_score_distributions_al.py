import argparse
import os
import sys

sys.path.append(os.getcwd())
from collections import defaultdict
import statistics

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import csv
from tqdm import tqdm
import seaborn as sns
import scipy
#from statsmodels.stats.weightstats import ztest
from matchmaker.dataloaders.independent_training_loader import *
from matchmaker.dataloaders.id_sequence_loader import *
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.data.fields.text_field import Token
from allennlp.common import Params, Tqdm
Tqdm.default_mininterval = 1
from matchmaker.dataloaders.bling_fire_tokenizer import BlingFireTokenizer
from matchmaker.autolabel_domain.compare_queries import read_collection

def read_qrels(qrels_file_path):
    qrels = {}
    with open(qrels_file_path, 'r') as qrels_file:
        lines = qrels_file.readlines()

        for line in lines:
            line = line.split('\t')
            query_id = line[0]
            doc_id = line[2]
            rel_grade = int(line[3])
            if rel_grade > 0:
                if qrels.get(query_id):
                    qrels.get(query_id).append(doc_id)
                else:
                    qrels.update({query_id: [doc_id]})
    return qrels


def plot_rel_irrel(data_paths, qrels):
    #
    # work
    #
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams.update({'font.size': 20})
    plt.rc('legend', **{'fontsize': 14})
    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['svg.fonttype'] = "none"
    mpl.rcParams['grid.linewidth'] = 1.5

    mpl.rcParams['lines.linewidth'] = 2.5

    # fig, ax = plt.subplots()
    fig = plt.figure(figsize=(7.8, 7.8))
    ax = fig.add_subplot(1, 1, 1)

    colors = [
        # ("dimgray","black"),
        ("indianred", "firebrick"),
        # ("sandybrown","darkorange"),
        ("steelblue", "darkslateblue"),
        ("mediumseagreen", "seagreen"),
        ("mediumorchid", "purple"),
        ("darkturquoise", "darkcyan"),
        ("yellow", "yellow"),
        ("pink", "pink")]

    markers = [
        "o",
        "v",
        "^",
        "s",
        "p",
    ]

    ax.set_clip_on(False)
    ax.yaxis.grid(linestyle=":", zorder=0)

    data_rel = {x: {} for x in data_paths.keys()}

    for name, dp in data_paths.items():
        with open(dp, "r", encoding="utf8") as in_file:
            for i, line in tqdm(enumerate(in_file)):
                line = line.split("\t")

                if len(line) > 1:
                    if data_rel[name].get(line[0]):
                        data_rel[name].get(line[0]).append((line[1], float(line[3].strip('\n'))))
                    else:
                        data_rel[name].update({line[0]: [(line[1], float(line[3].strip('\n')))]})

    # only filter the relevant ones
    data_rel_filtered = {x: [] for x in data_paths.keys()}
    for name, ranking in data_rel.items():
        for query_id, docs in ranking.items():
            for doc in docs:
                if qrels.get(query_id):
                    if doc[0] in qrels.get(query_id):
                        data_rel_filtered[name].append(doc[1])

    data_irrel_filtered = {x: [] for x in data_paths.keys()}
    for name, ranking in data_rel.items():
        for query_id, docs in ranking.items():
            for doc in docs:
                if qrels.get(query_id):
                    if not doc[0] in qrels.get(query_id):
                        data_irrel_filtered[name].append(doc[1])


    all_margins = []
    all_margin_means = []
    all_lables = []
    all_color0 = []
    all_color1 = []

    for i, (label, margins) in enumerate(data_rel_filtered.items()):
        margins = np.array(margins)
        print(label, scipy.stats.shapiro(margins))

        # margins =  (margins - margins.mean()) / margins.std()

        all_margins.append(margins)
        #all_margin_means.append(margins.mean())
        print(len(margins))
        all_lables.append(label)
        all_color0.append(colors[i][0])
        all_color1.append(colors[i][1])

    #print(all_margin_means)

    #average_mean = sum(all_margin_means) / len(all_margin_means)

    #print(average_mean)

    for i, (label, margins) in enumerate(data_rel_filtered.items()):

        pos_s = np.array(margins) #[:, 0]  # * (overall_pos_avg / pos_scores_avg[label])
        # neg_s = np.array(margins)[:, 1]  # * (overall_neg_avg / neg_scores_avg[label])
        if i == 0:
            # m = pos_s - neg_s
            sns.kdeplot(pos_s, label=str(label) + '_rel', shade=True, color="black", linestyle=":", clip=(-5, 1200))
        else:
            # sns.kdeplot(np.abs(m-(pos_s-neg_s)),label=label, shade=True,color=colors[i][1],clip=(-5,10))
            sns.kdeplot(pos_s, label=str(label) + '_rel', shade=True, color=colors[i][1], clip=(-5, 1200))
            # ax.axvline(np.mean(np.abs(m-(pos_s-neg_s))),color=colors[i][1],linestyle=":")
            # print(label,"mean",np.mean(np.abs(m-(pos_s-neg_s))))

    for i, (label, margins) in enumerate(data_irrel_filtered.items()):

        pos_s = np.array(margins) #[:, 0]  # * (overall_pos_avg / pos_scores_avg[label])
        # neg_s = np.array(margins)[:, 1]  # * (overall_neg_avg / neg_scores_avg[label])
        if i == 0:
            # m = pos_s - neg_s
            sns.kdeplot(pos_s, label=str(label) + '_irrel', shade=True, color="black", linestyle=":", clip=(-5, 1200))
        else:
            # sns.kdeplot(np.abs(m-(pos_s-neg_s)),label=label, shade=True,color=colors[i][1],clip=(-5,10))
            sns.kdeplot(pos_s, label=str(label) + '_irrel', shade=True, color=colors[i+len(data_rel_filtered)][1], clip=(-5, 1200))
            # ax.axvline(np.mean(np.abs(m-(pos_s-neg_s))),color=colors[i][1],linestyle=":")
            # print(label,"mean",np.mean(np.abs(m-(pos_s-neg_s))))


    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_ylabel("Rel. Occurence")
    ax.set_xlabel("Score distribution", labelpad=7)

    ax.margins(0.01)

    # if model ==  "conv_knrm":
    ##    plt.legend(loc=(0.05,0.01),framealpha=1,markerscale=0,labelspacing=0.1)
    # else:
    # plt.legend(loc= (0.08,0.78) ,framealpha=1,labelspacing=0.1)
    plt.legend(loc="upper right", framealpha=1, labelspacing=0.1)
    fig.tight_layout()

    plt.savefig(os.path.join(output_dir, "score_dis_rel_irrel.svg"), bbox_inches='tight')

def plot_score_dis(data_paths):
    data = {x: [] for x in data_paths.keys()}

    for name, dp in data_paths.items():
        with open(dp, "r", encoding="utf8") as in_file:
            for i, line in tqdm(enumerate(in_file)):
                line = line.split("\t")

                if len(line) > 1:
                    data[name].append(float(line[3].strip('\n')))

    # now only plot the score of the positive document!

    #
    # work
    #
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams.update({'font.size': 20})
    plt.rc('legend', **{'fontsize': 14})
    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['svg.fonttype'] = "none"
    mpl.rcParams['grid.linewidth'] = 1.5

    mpl.rcParams['lines.linewidth'] = 2.5

    # fig, ax = plt.subplots()
    fig = plt.figure(figsize=(7.8, 7.8))
    ax = fig.add_subplot(1, 1, 1)

    colors = [
        # ("dimgray","black"),
        ("indianred", "firebrick"),
        # ("sandybrown","darkorange"),
        ("steelblue", "darkslateblue"),
        ("mediumseagreen", "seagreen"),
        ("mediumorchid", "purple"),
        ("darkturquoise", "darkcyan"),
        ("yellow", "yellow"),
        ("pink", "pink")]

    markers = [
        "o",
        "v",
        "^",
        "s",
        "p",
    ]

    ax.set_clip_on(False)
    ax.yaxis.grid(linestyle=":", zorder=0)

    all_margins = []
    all_margin_means = []
    all_lables = []
    all_color0 = []
    all_color1 = []

    for i, (label, margins) in enumerate(data.items()):
        margins = np.array(margins)
        print(label, scipy.stats.shapiro(margins))

        # margins =  (margins - margins.mean()) / margins.std()

        all_margins.append(margins)
        # all_margin_means.append(margins.mean())
        print(len(margins))
        all_lables.append(label)
        all_color0.append(colors[i][0])
        all_color1.append(colors[i][1])

    # print(all_margin_means)

    # average_mean = sum(all_margin_means) / len(all_margin_means)

    # print(average_mean)

    for i, (label, margins) in enumerate(data.items()):

        pos_s = np.array(margins)  # [:, 0]  # * (overall_pos_avg / pos_scores_avg[label])
        # neg_s = np.array(margins)[:, 1]  # * (overall_neg_avg / neg_scores_avg[label])
        if i == 0:
            # m = pos_s - neg_s
            sns.kdeplot(pos_s, label=label, shade=True, color="black", linestyle=":", clip=(-5, 1200))
        else:
            # sns.kdeplot(np.abs(m-(pos_s-neg_s)),label=label, shade=True,color=colors[i][1],clip=(-5,10))
            sns.kdeplot(pos_s, label=label, shade=True, color=colors[i][1], clip=(-5, 1200))
            # ax.axvline(np.mean(np.abs(m-(pos_s-neg_s))),color=colors[i][1],linestyle=":")
            # print(label,"mean",np.mean(np.abs(m-(pos_s-neg_s))))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_ylabel("Rel. Occurence")
    ax.set_xlabel("Score distribution", labelpad=7)

    ax.margins(0.01)

    # if model ==  "conv_knrm":
    ##    plt.legend(loc=(0.05,0.01),framealpha=1,markerscale=0,labelspacing=0.1)
    # else:
    # plt.legend(loc= (0.08,0.78) ,framealpha=1,labelspacing=0.1)
    plt.legend(loc="upper right", framealpha=1, labelspacing=0.1)
    fig.tight_layout()

    plt.savefig(os.path.join(output_dir, "score_dis.svg"), bbox_inches='tight')


if __name__ == '__main__':
    output_dir = "/newstorage5/salthamm/autolabel-experiments/plots"

    data_paths = {
        "DR":"/newstorage5/salthamm/msmarco/experiments/subset_dr_incremental/2022-09-20_0624_dr_train_subset_03_20000/test_test_top1000_bm25-output.txt",
        "BERT_CAT":"/newstorage5/salthamm/msmarco/experiments/subset_bertcat_incremental/2022-09-19_2151_dr_train_subset_03_50000/test_test_top1000_bm25-output.txt",
        "ColBERT": "/newstorage5/salthamm/msmarco/experiments/subset_colbert_incremental/2022-10-13_2126_dr_train_subset_03_50000/test_test_top1000_bm25-output.txt"
        }

    qrels_file = "/newstorage5/salthamm/msmarco/data/qrels.train.tsv"
    qrels = read_qrels(qrels_file)

    plot_score_dis(data_paths)

    plot_rel_irrel(data_paths, qrels)
