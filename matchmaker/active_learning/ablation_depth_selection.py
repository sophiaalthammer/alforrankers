import argparse
import os
import io
import selectors
import yaml
import subprocess
from typing import Dict, Tuple, List
import warnings
import gc
import time
from contextlib import nullcontext
import sys, traceback

os.environ['PYTHONHASHSEED'] = "42"  # very important to keep set operations deterministic
from transformers import logging

logging.set_verbosity_warning()

sys.path.append(os.getcwd())
import itertools

from allennlp.common import Params, Tqdm

import torch
import torch.distributed as dist
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch import nn as nn

import numpy
import random
import transformers

from allennlp.nn.util import move_to_device
from matchmaker.utils.utils import *
from matchmaker.utils.config import *
from matchmaker.eval import *

from rich.console import Console
console = Console()

from matchmaker.al_pipe import execute_and_return_run_folder, qbc_selection, diversity_selection, uncertainty_selection


if __name__ == '__main__':
    #
    # config
    #
    args = get_parser().parse_args()
    from_scratch = True
    train_mode = "Train"
    commitee_members = None
    if args.continue_folder:
        train_mode = "Evaluate"
        from_scratch = False
        run_folder = args.continue_folder
        config = get_config_single(os.path.join(run_folder, "config.yaml"), args.config_overwrites)
        epochs = config.get('epochs_ablation', None)
        epoch_folders = config.get('epoch_folders', None)
        commitee_members = config.get('commitee_members', None)
        print('these are the model epochs that are evaluated: {}'.format(epochs))
    else:
        raise Exception('need a continuation folder')
        #if not args.run_name:
        #    raise Exception("--run-name must be set (or continue-folder)")
        #config = get_config(args.config_file, args.config_overwrites)
        #run_folder = prepare_experiment(args, config)

    #
    # random seeds
    #
    torch.manual_seed(config["random_seed"])
    numpy.random.seed(config["random_seed"])
    random.seed(config["random_seed"])

    logger.info("Torch seed: %i ",torch.initial_seed())

    # hardcode gpu usage
    cuda_device = 0 # always take the first -> set others via cuda flag in bash
    perf_monitor = PerformanceMonitor.get()
    perf_monitor.start_block("startup")


    al_strategy = config['al_strategy']
    no_iterations = config.get('no_iterations_al', 10)
    if al_strategy == 'qbc':
        # now the for loop starts to go over the iterations of the qbc algorithm!
        no_comittee = config.get('no_comittee', 2)
    elif al_strategy == 'uncertainty':
        uncertainty_cutoff = config.get('uncertainty_cutoff', 0.5)

    random.seed(config['random_seed'])

    first_selection = config.get('first_selection', 'random')

    train_file_original = config['train_tsv']
    print('this is the original train file {}'.format(train_file_original))

    current_train_size = int(run_folder.split('/')[-1].split('_')[-1])
    print('this is the current train size {}'.format(current_train_size))
    if epochs:
        for epoch in epochs:
            print('start with epoch {}'.format(epoch))
            if epoch_folders:
                for epoch_folder in epoch_folders:
                    if epoch_folder.endswith('epoch_'+str(epoch)):
                        run_folder_first = str(epoch_folder)
                        print('found epoch folder {}'.format(epoch_folder))
                    # else:
                    #     # do inference with the model on the train files
                    #     config_overwrites_text = "warmstart_model_path: {}, " \
                    #                              "expirement_base_path: {}".format(os.path.join(run_folder,
                    #                              'epoch_{}-best-model.pytorch-state-dict'.format(epoch)),
                    #                              os.path.join(run_folder, 'ablation_depth_selection'))
                    #
                    #     run_folder_first = execute_and_return_run_folder(
                    #         ["python", "matchmaker/rerank.py", "--config-file",
                    #          args.config_file[0], args.config_file[1], args.config_file[2],
                    #          "--config-overwrites",
                    #          config_overwrites_text,
                    #          "--run-name", "epoch_{}".format(epoch)])

                    #folder_epoch_run = os.path.join(run_folder, 'ablation_depth_selection', "epoch_{}".format(epoch))
                        train_file_path_subset = os.path.join(run_folder_first, 'triples_added_epoch_{}.tsv'.format(epoch))

                        # then do the selection
                        if al_strategy == 'qbc':
                            print('do qbc based al')
                            committee_folders = []

                            if commitee_members:
                                committee_folders = [str(commitee_members)]
                                committee_folders.append(run_folder_first)
                            else:
                                raise Exception('i need a committee member to compare with')

                            current_train_size = qbc_selection(committee_folders, current_train_size,
                                                               run_folder_first, int(epoch), config,
                                                               train_file_path_subset, train_file_original)

                        elif al_strategy == 'diversity' or al_strategy == 'uncertainty':
                            if al_strategy == 'uncertainty':
                                print('do uncertainy based al')
                                selection_name = 'uncert'
                            else:
                                print('sampling by diversity')
                                selection_name = 'divers'

                            if al_strategy == 'uncertainty':
                                current_train_size, no_added_queries_2 = uncertainty_selection(run_folder_first, current_train_size,
                                                                                               config, train_file_path_subset,
                                                                                               train_file_original, run_folder,
                                                                                               epoch)
                            elif al_strategy == 'diversity':
                                current_train_size = diversity_selection(run_folder_first, current_train_size, config,
                                                                         train_file_path_subset, train_file_original, run_folder)

    # print('finished with the sub epochs, start with final model')
    # # the same with the final best model
    # run_folder_first = run_folder
    # epoch = 200
    # train_file_path_subset = os.path.join(run_folder, 'ablation_depth_selection' ,'triples_added_epoch_{}.tsv'.format(epoch))
    #
    # # then do the selection
    # if al_strategy == 'qbc':
    #     print('do qbc based al')
    #     committee_folders = []
    #
    #     if commitee_members:
    #         committee_folders = commitee_members
    #         committee_folders.append(run_folder_first)
    #     else:
    #         raise Exception('i need a committee member to compare with')
    #
    #     current_train_size = qbc_selection(committee_folders, current_train_size,
    #                                        run_folder_first, int(epoch), config,
    #                                        train_file_path_subset, train_file_original)
    #
    # elif al_strategy == 'diversity' or al_strategy == 'uncertainty':
    #     if al_strategy == 'uncertainty':
    #         print('do uncertainy based al')
    #         selection_name = 'uncert'
    #     else:
    #         print('sampling by diversity')
    #         selection_name = 'divers'
    #
    #     if al_strategy == 'uncertainty':
    #         current_train_size, no_added_queries_2 = uncertainty_selection(run_folder_first,
    #                                                                        current_train_size,
    #                                                                        config, train_file_path_subset,
    #                                                                        train_file_original, run_folder,
    #                                                                        epoch)
    #     elif al_strategy == 'diversity':
    #         current_train_size = diversity_selection(run_folder_first, current_train_size, config,
    #                                                  train_file_path_subset, train_file_original,
    #                                                  run_folder)



