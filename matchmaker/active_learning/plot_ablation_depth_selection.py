import argparse
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pickle
import yaml
from yaml.loader import SafeLoader
from collections import OrderedDict
import matplotlib.ticker as mtick

def read_queries_per_epoch(data_path):
    queries_per_epoch = {}
    # open the actually added queries at 15 epochs
    run_folder_original = '/'.join(data_path.split('/')[:-2])
    traindata_size = data_path.split('/')[-1].split('_')[-1]

    mapping_train_size_iteration = {'10000': 1, '5000': 0, '15000': 2, '20000': 3}
    with open(os.path.join(run_folder_original,
                           'added_queries_iteration{}.tsv'.format(mapping_train_size_iteration.get(traindata_size))),
              'r') as qfile:
        lines = qfile.readlines()
        query_ids = [line.strip('\n') for line in lines]
        queries_per_epoch.update({15: query_ids})

    run_folder_ablation = os.path.join(data_path, 'ablation_depth_selection')
    with open(os.path.join(run_folder_ablation, 'added_queries_iteration200.tsv'), 'r') as qfile:
        lines = qfile.readlines()
        query_ids = [line.strip('\n') for line in lines]
        queries_per_epoch.update({200: query_ids})

    list_folders = os.listdir(run_folder_ablation)
    for folder in list_folders:
        if os.path.isdir(folder):
            epoch = folder.strip('_')[-1]
            with open(os.path.join(run_folder_ablation, folder, 'added_queries_iteration{}.tsv'.format(epoch)),
                      'r') as qfile:
                lines = qfile.readlines()
                query_ids = [line.strip('\n') for line in lines]
                queries_per_epoch.update({int(epoch): query_ids})
    return queries_per_epoch

def compute_overlaps(data_paths):
    queries = {}
    for path_name, data_path in data_paths:
        queries_per_epoch = read_queries_per_epoch(data_path)

        # compute the overlap between the first
        queries_orig = set(queries_per_epoch.get(15))
        length_queries = len(queries_per_epoch.get(15))
        overlaps = {}
        for epoch, queries in queries_per_epoch.items():
            set(queries).intersection(queries_orig)
            overlaps.update({epoch: len(set(queries).intersection(queries_orig)) / length_queries})

        overlaps_sorted = OrderedDict(sorted(dict.items()))

        queries.update({path_name: overlaps_sorted})
    return overlaps_sorted


# first load the data for all data_paths

# then plot multiple

out_path = '/newstorage5/salthamm/msmarco/experiments/al/qbc/'
data_paths = {
    'qbc_5k': '/newstorage5/salthamm/msmarco/experiments/al/qbc/2022-11-22_1523_dr_qbc_iter20_add5000/final_trainings/2022-12-17_0723_dr_train_member_01_0_10000',
}
model_name = 'DR'

data_path = '/newstorage5/salthamm/msmarco/experiments/al/qbc/2022-11-22_1523_dr_qbc_iter20_add5000/final_trainings/2022-12-17_0723_dr_train_member_01_0_10000'

overlaps = compute_overlaps(data_paths)

print(overlaps)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
plt.xlabel('Epochs', fontsize=15)
plt.grid(True, linewidth=0.5, color='lightgrey', linestyle='-')
plt.ylabel('Overlap with selected train samples', fontsize=15)

i = 0
colours = sns.color_palette("Paired", 12).as_hex()

for path_name, overlaps_scores in overlaps.items():

    xs, ys = zip(*overlaps_scores.items())
    print(xs)
    print(ys)
    # display
    #plt.scatter(xs, ys, c=colours[i], marker='o', edgecolors=colours[i])
    bars = plt.bar(xs, ys, colour='green', width=1.0, label=None, alpha=0.2)
    plt.bar(xs, 1-ys, colour='red', width=1.0, bottom=ys, label=None, alpha=0.2)

    for bar in bars:
        x, y = bar.get_xy()
        w, h = bar.get_width(), bar.get_height()
        #ax.plot([x, x], [y, y + h], color='black', lw=2)
        ax.plot([x, x + w], [y + h, y + h], color=colours[i], lw=2, label=path_name)
        #ax.plot([x + w, x + w], [y, y + h], color='black', lw=4)
    #ax.margins(x=0.02)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc='lower right')  # path_name  #labels[::-1]
ax.patch.set_facecolor('white')
#ax.set_yticks(np.arange(0, 101, 25))
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.savefig(os.path.join(out_path, 'epochs_ablation_depth.svg'), bbox_inches='tight')





