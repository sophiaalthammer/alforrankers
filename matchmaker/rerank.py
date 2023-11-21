#
# train a neural-ir model
# -------------------------------

from typing import Dict, Tuple, List
import os
import warnings
import gc
import time
from contextlib import nullcontext
import sys, traceback

os.environ['PYTHONHASHSEED'] = "42"  # very important to keep set operations deterministic
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # needed because of the scann library
# try:
#    from grad_cache import GradCache
#    _grad_cache_available = True
# except ModuleNotFoundError:
#    _grad_cache_available = False
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
from matchmaker.active_learning.generate_training_subset import generate_train_subset, generate_train_subset_from_train

from matchmaker.utils.cross_experiment_cache import *
from matchmaker.utils.input_pipeline import *
from matchmaker.utils.performance_monitor import *
from matchmaker.eval import *
from torch.utils.tensorboard import SummaryWriter

from rich.console import Console
from rich.live import Live

console = Console()

if __name__ == "__main__":

    #
    # config
    #
    args = get_parser().parse_args()
    from_scratch = True
    train_mode = "Train"
    if args.continue_folder:
        train_mode = "Evaluate"
        from_scratch = False
        run_folder = args.continue_folder
        config = get_config_single(os.path.join(run_folder, "config.yaml"), args.config_overwrites)
    else:
        if not args.run_name:
            raise Exception("--run-name must be set (or continue-folder)")
        config = get_config(args.config_file, args.config_overwrites)
        run_folder = prepare_experiment(args, config)

    logger = get_logger_to_file(run_folder, "main")
    logger.info("Running: %s", str(sys.argv))
    tb_writer = SummaryWriter(run_folder)
    print_hello(config, run_folder, train_mode)

    #
    # random seeds
    #
    torch.manual_seed(config["random_seed"])
    numpy.random.seed(config["random_seed"])
    random.seed(config["random_seed"])

    logger.info("Torch seed: %i ", torch.initial_seed())

    # hardcode gpu usage
    cuda_device = 0  # always take the first -> set others via cuda flag in bash
    perf_monitor = PerformanceMonitor.get()
    perf_monitor.start_block("startup")

    #
    # create (and load) model instance
    # -------------------------------
    #

    word_embedder, padding_idx = get_word_embedder(config)
    model, encoder_type = get_model(config, word_embedder, padding_idx)
    model = build_model(model, encoder_type, word_embedder, config)
    model = model.cuda()

    #
    # warmstart model
    #
    if "warmstart_model_path" in config:
        load_result = model.load_state_dict(torch.load(config["warmstart_model_path"]), strict=False)
        logger.info('Warmstart init model from:  %s', config["warmstart_model_path"])
        logger.info(load_result)
        console.log("[Startup]", "Trained model loaded locally; result:", load_result)

    logger.info('Model %s total parameters: %s', config["model"],
                sum(p.numel() for p in model.parameters() if p.requires_grad))
    logger.info('Network: %s', model)

    #
    # setup-multi gpu training
    #
    is_distributed = False
    if torch.cuda.device_count() > 1:
        console.log("[Startup]","Let's use", torch.cuda.device_count(), "GPUs!")
        device_list = list(range(torch.cuda.device_count()))
        model = nn.DataParallel(model,device_list)
        is_distributed = True
    perf_monitor.set_gpu_info(torch.cuda.device_count(),torch.cuda.get_device_name())
    use_fp16 = config["use_fp16"]

    #
    # evaluate the end validation & test & leaderboard sets with the best model checkpoint
    #
    print("Done with training! Reloading best checkpoint ...")
    # print("Mem allocated:",torch.cuda.memory_allocated())

    if is_distributed:
        model_cpu = model.module.cpu()  # we need this strange back and forth copy for models > 1/2 gpu memory, because load_state copies the state dict temporarily
    else:
        model_cpu = model.cpu()  # we need this strange back and forth copy for models > 1/2 gpu memory, because load_state copies the state dict temporarily

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(10)  # just in case the gpu has not cleaned up the memory
    torch.cuda.reset_peak_memory_stats()
    #model_cpu.load_state_dict(torch.load(best_model_store_path, map_location="cpu"), strict=False)
    model = model_cpu.cuda(cuda_device)
    if is_distributed:
        model = nn.DataParallel(model)

    print("Model reloaded ! memory allocation:", torch.cuda.memory_allocated())

    best_validation_end_metrics = {}
    if "validation_end" in config:
        for validation_end_name, validation_end_config in config["validation_end"].items():
            print("Evaluating validation_end." + validation_end_name)

            validation_end_candidate_set = None
            if "candidate_set_path" in validation_end_config:
                validation_end_candidate_set = parse_candidate_set(validation_end_config["candidate_set_path"],
                                                                   validation_end_config["candidate_set_from_to"][
                                                                       1])
            best_metric, _, validated_count, _ = validate_model(validation_end_name, model, config,
                                                                validation_end_config,
                                                                run_folder, logger, cuda_device,
                                                                candidate_set=validation_end_candidate_set,
                                                                output_secondary_output=validation_end_config[
                                                                    "save_secondary_output"],
                                                                is_distributed=is_distributed)
            save_best_info(os.path.join(run_folder, "val-" + validation_end_name + "-info.csv"), best_metric)
            best_validation_end_metrics[validation_end_name] = best_metric

    if "test" in config:
        for test_name, test_config in config["test"].items():
            print("Evaluating test." + test_name)
            cs_at_n_test = None
            test_candidate_set = None
            if "candidate_set_path" in test_config:
                cs_at_n_test = best_validation_end_metrics[test_name]["cs@n"]
                test_candidate_set = parse_candidate_set(test_config["candidate_set_path"],
                                                         test_config["candidate_set_max"])
            test_result = test_model(model, config, test_config, run_folder, logger, cuda_device,
                                     "test_" + test_name,
                                     test_candidate_set, cs_at_n_test,
                                     output_secondary_output=test_config["save_secondary_output"],
                                     is_distributed=is_distributed)

    if "leaderboard" in config:
        for test_name, test_config in config["leaderboard"].items():
            print("Evaluating leaderboard." + test_name)
            test_model(model, config, test_config, run_folder, logger, cuda_device, "leaderboard" + test_name,
                       output_secondary_output=test_config["save_secondary_output"], is_distributed=is_distributed)

    perf_monitor.log_value("eval_gpu_mem", torch.cuda.memory_allocated() / float(1e9))
    perf_monitor.log_value("eval_gpu_mem_max", torch.cuda.max_memory_allocated() / float(1e9))
    perf_monitor.log_value("eval_gpu_cache", torch.cuda.memory_reserved() / float(1e9))
    perf_monitor.log_value("eval_gpu_cache_max", torch.cuda.max_memory_reserved() / float(1e9))
    torch.cuda.reset_peak_memory_stats()

    perf_monitor.save_summary(os.path.join(run_folder, "efficiency-metrics.json"))

    if config.get("run_dense_retrieval_eval", False):
        print("Starting dense_retrieval")
        import sys, subprocess

        if config["model"] == "ColBERTer" or config["model"] == "CoColBERTer" or config["model"] == "CoCoColBERTer":
            subprocess.Popen(["python", "matchmaker/colberter_retrieval.py", "encode+index+search", "--config",
                              config["colberter_retrieval_config"]
                                 , "--run-name", args.run_name, "--config-overwrites",
                              "trained_model: " + run_folder])
            print(run_folder)
        else:
            subprocess.Popen(["python", "matchmaker/dense_retrieval.py", "encode+index+search", "--config",
                              config["dense_retrieval_config"]
                                 , "--run-name", args.run_name, "--config-overwrites",
                              "trained_model: " + run_folder])
            print(run_folder)
    print(run_folder)

