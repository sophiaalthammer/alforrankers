import argparse
import os
import io
import selectors
#from matchmaker.utils.config import *
import yaml
import subprocess
from typing import Dict, Tuple, List
import warnings
import gc
import time
from contextlib import nullcontext
import sys, traceback

os.environ['PYTHONHASHSEED'] = "42"
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
from matchmaker.distillation.dynamic_teacher import DynamicTeacher
from matchmaker.utils.running_average import RunningAverage

from matchmaker.models.all import get_model, get_word_embedder, build_model
from matchmaker.losses.all import get_loss, merge_loss
from matchmaker.active_learning.generate_training_subset import *

from matchmaker.utils.cross_experiment_cache import *
from matchmaker.utils.input_pipeline import *
from matchmaker.utils.performance_monitor import *
from matchmaker.eval import *
from torch.utils.tensorboard import SummaryWriter

from itertools import combinations
from matchmaker.active_learning.selection_strategies_al import *
from matchmaker.active_learning.utils import *
from matchmaker.active_learning.generate_training_subset import *

from rich.console import Console
from rich.live import Live

console = Console()


if __name__ == '__main__':
    #
    # config
    #
    args = get_parser().parse_args()
    from_scratch = True
    train_mode = "Train"
    iteration_continue = None
    commitee_members = None
    continuation = False
    if args.continue_folder:
        train_mode = "Evaluate"
        from_scratch = False
        run_folder = args.continue_folder
        config = get_config_single(os.path.join(run_folder, "config.yaml"), args.config_overwrites)
        continuation = True
        iteration_continue = config.get('iteration_continue')
        commitee_members = config.get('commitee_members', None)
        do_rerank = config.get('do_rerank', None)
        print('these are the continuation commitee members: {}'.format(commitee_members))
    else:
        if not args.run_name:
            raise Exception("--run-name must be set (or continue-folder)")
        config = get_config(args.config_file, args.config_overwrites)
        run_folder = prepare_experiment(args, config)

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

    train_file_original = config['train_tsv']
    print('this is the original train file {}'.format(train_file_original))

    if continuation:
        start_iterations = iteration_continue
    else:
        start_iterations = 0

    first_selection = config.get('first_selection', 'random')
    do_final_training = config.get('continue_final_training', False)

    current_train_size = 0
    for iteration in range(start_iterations, no_iterations):
        print('start with iteration {}'.format(iteration))

        # create start subset either with random selection or with active learning selection
        if iteration == 0 and not continuation:
            if first_selection == 'random':
                print('initialization random')
                train_file_path_subset = generate_train_subset_from_train(config["train_tsv"], run_folder,
                                                                          config['train_data_size'],
                                                                          config["random_seed"])
                config['train_tsv'] = train_file_path_subset
                current_train_size = config['train_data_size']

                print('created training subset of size {}'.format(config['train_data_size']))
            else:
                print('initialization not random but with selection')
                # inference with rerank.py to create the first train selection, start with that training file
                config_overwrites_text = "expirement_base_path: {}".format(run_folder)

                run_folder_first = execute_and_return_run_folder(
                    ["python", "matchmaker/rerank.py", "--config-file",
                     args.config_file[0], args.config_file[1], args.config_file[2],
                     "--config-overwrites",
                     config_overwrites_text,
                     "--run-name", "dr_first_selection_{}".format(config['train_data_size'])])
                train_file_path_subset = os.path.join(run_folder, 'triples.train.subset.{}.tsv'.format(config['train_data_size']))
                current_train_size = 0
                if al_strategy == 'qbc':
                    print('for qbc with only one member qbc selection equals random seleciton because every query will '
                          'have the same vote entropy, therefore do random selection')
                    train_file_path_subset = generate_train_subset_from_train(config["train_tsv"], run_folder,
                                                                              config['train_data_size'],
                                                                              config["random_seed"])
                    config['train_tsv'] = train_file_path_subset
                    current_train_size = config['train_data_size']

                    print('created training subset of size {}'.format(config['train_data_size']))
                elif al_strategy == 'uncertainty':
                    current_train_size, no_added_queries_2 = uncertainty_selection(run_folder_first,
                                                                                   current_train_size,
                                                                                   config, train_file_path_subset,
                                                                                   train_file_original, run_folder,
                                                                                   -1)
                elif al_strategy == 'diversity':
                    current_train_size = diversity_selection(run_folder_first, current_train_size, config,
                                                             train_file_path_subset, train_file_original, run_folder,
                                                             iteration)

                config['train_tsv'] = train_file_path_subset
                print('this is the current train size after initialization {}'.format(current_train_size))

        if iteration == start_iterations and continuation:
            if iteration == 0:
                current_train_size = config['train_data_size']
            else:
                current_train_size = config['train_data_size'] * iteration + config['train_data_size']
            train_file_path_subset = config['continue_train_file']
            config['train_tsv'] = train_file_path_subset
            print('this is the continuation train file {}'.format(config['train_tsv']))

        if al_strategy == 'qbc':
            print('do qbc based al')
            committee_folders = []

            if iteration == start_iterations and continuation and commitee_members:
                committee_folders = commitee_members
                config_start = get_config_single(os.path.join(committee_folders[0], 'config.yaml'))
                first_rand_seed = config_start.get('random_seed')
                if len(committee_folders) <= no_comittee:
                    for comittee_member in range(len(committee_folders), no_comittee):
                        print('start with training of comittee member {}'.format(comittee_member))
                        rand_seed = random.randrange(0, 100000)
                        if comittee_member == 0:
                            first_rand_seed = rand_seed

                        config_overwrites_text = "train_tsv: {},random_seed: {}, train_data_size: {}, " \
                                                 "train_subset: True, expirement_base_path: {}".format(
                            train_file_path_subset, rand_seed,
                            math.ceil(current_train_size * float(config['m_percentage_subset'])), run_folder)

                        print(config_overwrites_text)
                        run_folder_commitee_member = execute_and_return_run_folder(
                            ["python", "matchmaker/train.py", "--config-file",
                             args.config_file[0], args.config_file[1], args.config_file[2],
                             "--config-overwrites",
                             config_overwrites_text,
                             "--run-name", "dr_train_member_{:02d}_{}_{}".format(iteration, comittee_member, math.ceil(
                                current_train_size * float(config['m_percentage_subset'])))])
                        print('finished with training of comittee member {}'.format(comittee_member))
                        committee_folders.append(run_folder_commitee_member)

                if do_rerank:
                    print('do reranking first')
                    for committee_member in committee_folders:
                        run_folder_first = execute_and_return_run_folder(
                            ["python", "matchmaker/rerank.py", "--config-file",
                             args.config_file[0], args.config_file[1], args.config_file[2],
                             "--continue-folder", "{}".format(committee_member)])

            else:
                for comittee_member in range(no_comittee):
                   print('start with training of comittee member {}'.format(comittee_member))
                   rand_seed = random.randrange(0,100000)
                   if comittee_member == 0:
                       first_rand_seed = rand_seed

                   config_overwrites_text = "train_tsv: {},random_seed: {}, train_data_size: {}, train_subset: True, " \
                                            "expirement_base_path: {}".format(
                       train_file_path_subset, rand_seed, math.ceil(current_train_size*float(config['m_percentage_subset'])), run_folder)

                   run_folder_commitee_member = execute_and_return_run_folder(["python", "matchmaker/train.py", "--config-file",
                        args.config_file[0], args.config_file[1], args.config_file[2],
                        "--config-overwrites",
                        config_overwrites_text,
                        "--run-name", "dr_train_member_{:02d}_{}_{}".format(iteration, comittee_member,
                        math.ceil(current_train_size*float(config['m_percentage_subset'])))])
                   print('finished with training of comittee member {}'.format(comittee_member))
                   committee_folders.append(run_folder_commitee_member)

            if do_final_training:
                # take the best model as continuation, committee_folders[0]
                config_overwrites_text_final = "train_tsv: {},random_seed: {}, train_data_size: {}, train_subset: True, " \
                                         "expirement_base_path: {}, epochs: {}, warmstart_model_path: {}".format(
                    train_file_path_subset, first_rand_seed,
                    math.ceil(current_train_size * float(config['m_percentage_subset'])),
                    os.path.join(run_folder, 'final_trainings'),
                    int(config.get('epochs_final_train', 200))-int(config['batch_size_train']),
                    os.path.join(committee_folders[0], 'final-model.pytorch-state-dict'))
                print('start with final training')
                run_folder_final = execute_and_return_run_folder(
                    ["python", "matchmaker/train.py", "--config-file",
                     args.config_file[0], args.config_file[1], args.config_file[2],
                     "--config-overwrites",
                     config_overwrites_text_final,
                     "--run-name", "dr_train_member_{:02d}_{}_{}".format(iteration, 0, math.ceil(
                        current_train_size * float(config['m_percentage_subset'])))])
                print('finished with final training {}'.format(run_folder_final))

            current_train_size = qbc_selection(committee_folders, current_train_size, run_folder, iteration, config,
                                               train_file_path_subset, train_file_original)

        elif al_strategy == 'diversity' or al_strategy == 'uncertainty':
            if al_strategy == 'uncertainty':
                print('do uncertainy based al')
                selection_name = 'uncert'
            else:
                print('sampling by diversity')
                selection_name = 'divers'

            if continuation and iteration == start_iterations and commitee_members:
                run_folder_iteration = commitee_members
                config_start = get_config_single(os.path.join(run_folder_iteration, 'config.yaml'))
                rand_seed = config_start.get('random_seed')
            else:
                print('start with training of model at iteration {}'.format(iteration))
                rand_seed = random.randrange(0, 100000)
                config_overwrites_text = "train_tsv: {},random_seed: {}, train_subset: False, " \
                                         "expirement_base_path: {}".format(train_file_path_subset, rand_seed, run_folder)

                run_folder_iteration = execute_and_return_run_folder(
                    ["python", "matchmaker/train.py", "--config-file",
                     args.config_file[0], args.config_file[1], args.config_file[2],
                     "--config-overwrites",
                     config_overwrites_text,
                     "--run-name", "dr_train_{}_{:02d}_{}".format(selection_name, iteration, current_train_size)])
                print('finished with training of {} iteration {}'.format(selection_name, iteration))

            if do_rerank:
                print('do reranking first')
                run_folder_first = execute_and_return_run_folder(
                    ["python", "matchmaker/rerank.py", "--config-file",
                     args.config_file[0], args.config_file[1], args.config_file[2],
                     "--continue-folder", "{}".format(run_folder_iteration)])

            if do_final_training:
                # take the best model as continuation, committee_folders[0]
                config_overwrites_text_final = "train_tsv: {},random_seed: {}, train_subset: False, " \
                                               "expirement_base_path: {}, epochs: {}, warmstart_model_path: {}".format(
                    train_file_path_subset, rand_seed, os.path.join(run_folder, 'final_trainings'),
                    int(config.get('epochs_final_train', 200))-int(config['batch_size_train']),
                    os.path.join(run_folder_iteration, 'final-model.pytorch-state-dict'))
                print('start with final training')
                run_folder_final = execute_and_return_run_folder(
                    ["python", "matchmaker/train.py", "--config-file",
                     args.config_file[0], args.config_file[1], args.config_file[2],
                     "--config-overwrites",
                     config_overwrites_text_final,
                     "--run-name", "dr_train_{}_{:02d}_{}".format(selection_name, iteration, current_train_size)])
                print('finished with final training {}'.format(run_folder_final))

            if al_strategy == 'uncertainty':
                current_train_size, no_added_queries_2 = uncertainty_selection(run_folder_iteration, current_train_size,
                                                                               config, train_file_path_subset,
                                                                               train_file_original, run_folder,
                                                                               iteration)
            elif al_strategy == 'diversity':
                current_train_size = diversity_selection(run_folder_iteration, current_train_size, config,
                                                         train_file_path_subset, train_file_original, run_folder,
                                                         iteration)


    # after iterations do final training
    print('finished with the iterations do final training')

    config_overwrites_text = "train_tsv: {}, train_subset: False, expirement_base_path: {}, epochs: {}".format(
        train_file_path_subset, os.path.join(run_folder, 'final_trainings'), config.get('epochs_final_train', 200))

    run_folder_commitee_member = execute_and_return_run_folder(["python", "matchmaker/train.py", "--config-file",
                                                                args.config_file[0], args.config_file[1],
                                                                args.config_file[2],
                                                                "--config-overwrites",
                                                                config_overwrites_text,
                                                                "--run-name",
                                                                "dr_final_modell_00_{}".format(current_train_size)])
    print('finished with training of final model')









