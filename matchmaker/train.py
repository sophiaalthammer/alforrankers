#
# train a neural-ir model
# -------------------------------

from typing import Dict, Tuple, List
import os
import warnings
import gc
import time
from contextlib import nullcontext
import sys,traceback
os.environ['PYTHONHASHSEED'] = "42" # very important to keep set operations deterministic
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # needed because of the scann library
#try:
#    from grad_cache import GradCache
#    _grad_cache_available = True
#except ModuleNotFoundError:
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
from matchmaker.losses.all import get_loss,merge_loss
from matchmaker.active_learning.generate_training_subset import *
from matchmaker.autolabel_domain.robust04_nfoldcrosstrain import create_nfold_train_test_data

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
    print_hello(config,run_folder,train_mode)

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

    #
    # create the training subset in case of subset training
    # -------------------------------
    #
    train_subset = config.get("train_subset", False)
    train_subset_incrementally = config.get('train_subset_incrementally', False)
    train_subset_warmstart = config.get('train_subset_warmstart', False)
    train_subset_control_topic_no = config.get('train_subset_control_topic_no', -1)
    train_subset_firstk = config.get('train_subset_firstk', False)
    triplet_no_per_topic = config.get('triplet_no_per_topic', 10)
    if train_subset:
        if not train_subset_incrementally:
            if train_subset_control_topic_no > -1:
                train_file_path = generate_subset_control_topic_no(config["train_tsv"], run_folder, config['train_subset_control_topic_no'], config["random_seed"], triplet_no_per_topic)
            elif train_subset_firstk:
                train_file_path = generate_train_subset_from_train_firstk(config["train_tsv"], run_folder, config['train_data_size'])
            else:
                train_file_path = generate_train_subset_from_train(config["train_tsv"], run_folder, config['train_data_size'], config["random_seed"])
            config["train_tsv"] = train_file_path

        else:
            if not train_subset_warmstart:
                if train_subset_control_topic_no > -1:
                    train_file_path = generate_subset_control_incrementally(config["train_tsv"], run_folder, config['train_subset_control_topic_no'],
                                                                            config["random_seed"], triplet_no_per_topic,
                                                                            config['expirement_base_path'],
                                                                            config['previous_exp_name'])
                else:
                    train_file_path = generate_train_subset_incrementally(config["train_tsv"], run_folder, config['train_data_size'],
                                                                          config["random_seed"], config['expirement_base_path'], config['previous_exp_name'])
            else:
                if train_subset_control_topic_no > -1:
                    train_file_path, warmstart_path = generate_subset_control_incrementally(config["train_tsv"], run_folder, config['train_subset_control_topic_no'],
                                                                            config["random_seed"], triplet_no_per_topic,
                                                                            config['expirement_base_path'],
                                                                            config['previous_exp_name'], warmstart_model=True)
                else:
                    train_file_path, warmstart_path = generate_train_subset_incrementally(config["train_tsv"], run_folder, config['train_data_size'],
                                                                          config["random_seed"], config['expirement_base_path'], config['previous_exp_name'], warmstart_model=True)
                config["warmstart_model_path"] = warmstart_path
            config["train_tsv"] = train_file_path

    #
    # create the training nfold subset
    # -------------------------------
    #
    train_subset_nfold = config.get("train_subset_nfold", False)
    if train_subset_nfold:
        if config['nfold_sampling'] == 'nfold':
            print('use nfolds from {} to {}'.format(config["fold_lb"], config["fold_ub"]))
        else:
            print('use random sampling for fold')
        # now i use the config["train_tsv"] to subsample the size!
        create_nfold_train_test_data(config['collection'], config['queries'], run_folder, config['candidate_path'],
                                     config['validation_cont']['qrels'], sampling=config['nfold_sampling'],
                                     fold_lb=config.get("fold_lb", 0), fold_ub=config.get('fold_ub', 0),
                                     train_size=config.get('fold_train_size', 0), index_dir=config.get('index_dir', ''),
                                     n_samples_per_query=config.get('n_samples_per_query', -1))

        config["train_tsv"] = os.path.join(run_folder, 'train_triples_nfold.tsv')
        config["validation_cont"]["tsv"] = os.path.join(run_folder, 'test_nfold_queries_rerank.tsv')
        config["test"]["top1000_description"]["tsv"] = os.path.join(run_folder, 'test_nfold_queries.tsv')

    #
    # create (and load) model instance
    # -------------------------------
    #

    # load candidate set for efficient cs@N validation 
    validation_cont_candidate_set = None
    if from_scratch and "candidate_set_path" in config["validation_cont"]:
        validation_cont_candidate_set = parse_candidate_set(config["validation_cont"]["candidate_set_path"],config["validation_cont"]["candidate_set_from_to"][1])

    word_embedder, padding_idx = get_word_embedder(config)
    model, encoder_type = get_model(config,word_embedder,padding_idx)
    model = build_model(model,encoder_type,word_embedder,config)
    model = model.cuda()

    #
    # warmstart model 
    #
    if "warmstart_model_path" in config:
        print('load model from {}'.format(config["warmstart_model_path"]))
        load_result = model.load_state_dict(torch.load(config["warmstart_model_path"]),strict=False)
        logger.info('Warmstart init model from:  %s', config["warmstart_model_path"])
        logger.info(load_result)
        console.log("[Startup]","Trained model loaded locally; result:",load_result)

    logger.info('Model %s total parameters: %s', config["model"], sum(p.numel() for p in model.parameters() if p.requires_grad))
    logger.info('Network: %s', model)

    params_group0 = []
    params_group1 = []
    we_params = []

    for p_name,par in model.named_parameters():
        if par.requires_grad:
            group1_check = p_name
            spl=p_name.split(".")
            if len(spl) >= 2 and (spl[0] == 'neural_ir_model' or spl[0] == 'bert_model'):
                group1_check = spl[1]
            elif len(spl) >= 2:
                group1_check = spl[0]

            if not config["train_embedding"] and p_name.startswith("word_embedding"):
                pass
            elif config["train_embedding"] and p_name.startswith("word_embedding") and not "LayerNorm" in p_name and not config["use_fp16"]:
                we_params.append(par)
                logger.info("we_params: %s",p_name)
            elif group1_check in config["param_group1_names"]:
                params_group1.append(par)
                logger.info("params_group1: %s",p_name)
            else:
                params_group0.append(par)

    all_params =[
        {"params":params_group0,"lr":config["param_group0_learning_rate"],"weight_decay":config["param_group0_weight_decay"]},
        {"params":params_group1,"lr":config["param_group1_learning_rate"],"weight_decay":config["param_group1_weight_decay"]}
    ]

    embedding_optimizer=None
    use_embedding_optimizer = False
    if len(we_params) > 0 and config["train_embedding"] and not config["use_fp16"]:
        use_embedding_optimizer = True
        if config["embedding_optimizer"] == "adam":
            embedding_optimizer = Adam(we_params, lr=config["embedding_optimizer_learning_rate"],weight_decay=config["param_group0_weight_decay"])

        elif config["embedding_optimizer"] == "sparse_adam":
            embedding_optimizer = SparseAdam(we_params, lr=config["embedding_optimizer_learning_rate"])

        elif config["embedding_optimizer"] == "sgd":
            embedding_optimizer = SGD(we_params, lr=config["embedding_optimizer_learning_rate"],momentum=config["embedding_optimizer_momentum"])

    if config["optimizer"] == "adam":
        optimizer = Adam(all_params)

    elif config["optimizer"] == "sgd":
        optimizer = SGD(all_params, momentum=0.5)

    #lr_scheduler = ReduceLROnPlateau(optimizer, mode="max",
    #                                 patience=config["learning_rate_scheduler_patience"],
    #                                 factor=config["learning_rate_scheduler_factor"],
    #                                 verbose=True)

    lr_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,1,85_00)
    if embedding_optimizer is not None:
        lr_scheduler2 = transformers.get_cosine_schedule_with_warmup(embedding_optimizer,1,85_00)
    
    use_title_body_sep = config["use_title_body_sep"]
    use_cls_scoring = config["use_cls_scoring"]
    train_sparsity = config["minimize_sparsity"]
    #if train_sparsity:
    #    config["sparsity_log_path"] = os.path.join(run_folder, config["sparsity_log_path"])
    train_qa_spans = config["train_qa_spans"]
    use_in_batch_negatives = config["in_batch_negatives"]
    use_in_batch_queries = config['in_batch_queries']
    use_intra_batch_queries = config['intra_batch_queries']
    query_vecs_pos_prev_batch = None
    #query_vecs_pos_prev_batch.cuda()
    use_in_batch_queries_length = config['in_batch_queries_length']

    pretrain = config.get("pretrain", False)
    late_mlm = config.get("late_mlm", False)

    use_submodel_caching = "submodel_train_cache_path" in config
    if use_submodel_caching:
        submodel_cacher = CrossExperimentReplayCache(config["submodel_train_cache_path"],is_readonly=config["submodel_train_cache_readonly"])

    early_stopper = EarlyStopping(patience=config["early_stopping_patience"], mode="max")

    use_dynamic_teacher = config.get("dynamic_teacher",False)
    loss_avg_running = RunningAverage(1000)
    per_cluster_idx_diff = defaultdict(list)

    #
    # setup-multi gpu training 
    #
    is_distributed = False
    if torch.cuda.device_count() > 1:
        if use_dynamic_teacher:
            console.log("[Startup]","Let's use", torch.cuda.device_count()-1, "GPUs for training! and 1 GPU for dynamic teacher inference")
            device_list = list(range(torch.cuda.device_count()))[:-1]
            if len(device_list) > 1: # so in case we have a 1/1 split, don't actually parallelize the model
                model = nn.DataParallel(model,device_list)
                is_distributed = True
        else:
            console.log("[Startup]","Let's use", torch.cuda.device_count(), "GPUs!")
            device_list = list(range(torch.cuda.device_count()))
            model = nn.DataParallel(model,device_list)
            is_distributed = True
    perf_monitor.set_gpu_info(torch.cuda.device_count(),torch.cuda.get_device_name())
    use_fp16 = config["use_fp16"]

    #
    # setup loss
    #
    best_metric_info_file = os.path.join(run_folder, "best-info.csv")
    best_model_store_path = os.path.join(run_folder, "best-model.pytorch-state-dict")

    if from_scratch:
        ranking_loss_fn, qa_loss_fn, inbatch_loss_fn, use_list_loss,use_inbatch_list_loss, inbatch_loss_query, inbatch_loss_query_length = get_loss(config)
        ranking_loss_fn = ranking_loss_fn.cuda(cuda_device)
        train_pairwise_distillation = config["train_pairwise_distillation"]
        train_per_term_scores = config["dynamic_teacher_per_term_scores"]
        dual_loss = config.get("train_dual_loss", False)
        approximate_margin_factor_inb_negs = config.get("approximate_margin_factor_inb_negs", -1)
        
        teacher_pos_key = "pos_score"
        teacher_neg_key = "neg_score"
        if config["train_pairwise_distillation_on_passages"]:
            teacher_pos_key = "dyn_teacher_scores_pos" #"pos_score_passages"
            teacher_neg_key = "dyn_teacher_scores_neg" #"neg_score_passages"
        if config.get("dynamic_teacher_use_pair_scores",False):
            teacher_pos_key = "dyn_teacher_pair_scores_pos"
            teacher_neg_key = "dyn_teacher_pair_scores_neg"
        loss_file_path = os.path.join(run_folder, "training-loss.csv")
        # write csv header once
        with open(loss_file_path, "w") as loss_file:
            loss_file.write("sep=,\nEpoch,After_Batch,Loss\n")

    
        store_n_best_checkpoints = config["store_n_best_checkpoints"]

        min_steps_training = config["min_steps_training"]

    # keep track of the best metric
    if from_scratch:
        best_metric_info = {}
        best_metric_info["metrics"]={}
        best_metric_info["metrics"][config["validation_metric"]] = 0
    else:
        best_metric_info = read_best_info(best_metric_info_file)

    perf_monitor.stop_block("startup")
    
    # just test that it works (very much needed for new models ^^)
    if is_distributed:
        model.module.get_param_stats()
    else:
        model.get_param_stats()

    #if pretrain:
    #    vocab_size = model.vocab_size
    #    console.log("[Train]", "This is the vocab size of the model ", vocab_size)
    
    #
    # training / saving / validation loop
    # -------------------------------
    #

    scaler = torch.cuda.amp.GradScaler()

    gradient_cache = False
    if config.get("cache_chunk_size",-1) > -1:
        gradient_cache = True
        cache_chunk_size = config["cache_chunk_size"]
        if is_distributed:
            gc = GradCache(models=[model.module.bert_model],
                           # maybe i need also the compression layers for mlm, because i also use them
                           # in cocondenser training, but first try without tokenlevelpretraining
                           chunk_sizes=cache_chunk_size,
                           loss_fn=model.module.compute_contrastive_loss,
                           get_rep_fn=lambda x: x.hidden_states[-1][:, 0],
                           fp16=use_fp16,
                           scaler=scaler
                           )
        else:
            gc = GradCache(models=[model.bert_model],  # maybe i need also the compression layers for mlm, because i also use them
                                                    # in cocondenser training, but first try without tokenlevelpretraining
                    chunk_sizes=cache_chunk_size,
                    loss_fn=model.compute_contrastive_loss,
                    get_rep_fn=lambda x: x.hidden_states[-1][:, 0],
                    fp16=use_fp16,
                    scaler=scaler
                )
        per_device_train_batch_size = config["batch_size_train"]

    #@profile
    #def work():
    try:
        global_i = 0
        if from_scratch:
            for epoch in range(0, int(config["epochs"])):
                if early_stopper.stop:
                    break
                perf_monitor.start_block("train")
                perf_start_inst = 0


                input_loader = allennlp_triple_training_loader(config, config, config["train_tsv"])
                if use_dynamic_teacher:
                    input_loader = DynamicTeacher(config, input_loader,logger)
                else:
                    console.log("[Train]","Static training from: ",config["train_tsv"])


                #time.sleep(len(training_processes))  # fill the queue
                logger.info("[Epoch "+str(epoch)+"] --- Start training ")

                #
                # vars we need for the training loop 
                # -------------------------------
                #
                model.train()  # only has an effect, if we use dropout & regularization layers in the model definition...

                tensorboard_cats = ["PairwiseRankScore/Loss","PairwiseRankScore/Accuracy","PairwiseRankScore/pos_avg","PairwiseRankScore/neg_avg","PairwiseRankScore/score_diff","Gradients/GradNorm"]
                if dual_loss:
                    tensorboard_cats += ["PairwiseRankScore_Secondary/Loss","PairwiseRankScore_Secondary/Accuracy","PairwiseRankScore_Secondary/pos_avg",
                                         "PairwiseRankScore_Secondary/neg_avg","PairwiseRankScore_Secondary/score_diff"]
                if use_in_batch_negatives:
                    tensorboard_cats += ["In-Batch/Loss","In-Batch/Accuracy","In-Batch/score_diff","In-Batch/Teacher-Accuracy"]
                if use_in_batch_queries:
                    tensorboard_cats += ["In-Batch/QLoss", "In-Batch/QDistance"]
                if use_in_batch_queries_length:
                    tensorboard_cats += ["In-Batch/QLengthLoss", "In-Batch/QLength"]
                if use_intra_batch_queries:
                    tensorboard_cats += ["In-Batch/IntraQLoss", "In-Batch/IntraQDistance"]
                if train_qa_spans:
                    tensorboard_cats += ["QA/Loss","QA/Start_accuracy","QA/End_accuracy","QA/Answerability_pos_accuracy","QA/Answerability_neg_accuracy","QA/Loss_weighted_ranking","QA/Loss_weighted_qa"]
                if train_sparsity:
                    tensorboard_cats += ["Sparsity/loss","Sparsity/sparsity_avg_pos","Sparsity/sparsity_avg_neg"]
                if train_per_term_scores:
                    tensorboard_cats += ["PairwiseRankScore/PerTermLoss"]
                if use_fp16:
                    tensorboard_cats += ["Gradients/Scaler"]
                
                tensorboard_stats = {}
                for t in tensorboard_cats:
                    tensorboard_stats[t] = torch.zeros(1).cuda(cuda_device)


                training_batch_size = int(config["batch_size_train"])
                # label is always set to 1 - indicating first input is pos (see ranking_loss_fn:MarginRankingLoss) + cache on gpu
                label = torch.ones(training_batch_size).cuda(cuda_device)

                # helper vars for quick checking if we should validate during the epoch
                validate_every_n_batches = config["validate_every_n_batches"]
                do_validate_every_n_batches = validate_every_n_batches > -1

                validate_every_n_epochs = config.get("validate_every_n_epochs", 10000000)

                concated_sequences = config["token_embedder_type"] == "bert_cat"

                gradient_accumulation = False
                if config["gradient_accumulation_steps"] > 0 and not gradient_cache:
                    gradient_accumulation = True
                    gradient_accumulation_steps = config["gradient_accumulation_steps"]
                
                #
                # train loop 
                # -------------------------------
                #
                i=0
                lastRetryI=0
                
                with Live("",console=console,auto_refresh=False) as status:
                    for batch in input_loader:
                        if i == 0:
                            status.update("[bold magenta]           Starting...",refresh=True)
                        try:
                            batch = move_to_device(batch, cuda_device)

                            #
                            # create input 
                            #
                            pos_in = []
                            neg_in = []
                            if concated_sequences:
                                pos_in.append(batch["doc_pos_tokens"])  
                                neg_in.append(batch["doc_neg_tokens"])
                            elif pretrain and gradient_cache:
                                inputs = batch["seq_tokens"]
                                if torch.cuda.device_count() > 1:
                                    local_rank = torch.cuda.device_count()
                                else:
                                    local_rank = -1
                                #local_rank = config.get("cocondenser_local_rank", -1)

                                #print(inputs)
                                #print(cache_chunk_size)

                                # Construct the gradient cache
                                chunked_inputs = split_tensor_dict(inputs, cache_chunk_size=cache_chunk_size)
                                #print(chunked_inputs)
                                for c in chunked_inputs:
                                    c['output_hidden_states'] = True
                                if is_distributed:
                                    cls_hiddens, rnd_states = gc.forward_no_grad(model.module.bert_model, chunked_inputs)
                                    if model.module.use_retrieval_compression:
                                        with torch.no_grad():
                                            cls_hiddens = model.module.compressor_retrieval(cls_hiddens)
                                else:
                                    cls_hiddens, rnd_states = gc.forward_no_grad(model.bert_model, chunked_inputs)
                                    if model.use_retrieval_compression:
                                        with torch.no_grad():
                                            cls_hiddens = model.compressor_retrieval(cls_hiddens)
                                if local_rank > -1:
                                    cls_hiddens = gather_tensors(cls_hiddens.contiguous(), local_rank=local_rank)[0]
                                #print('cls hiddens')
                                #print(cls_hiddens)
                                #print(cls_hiddens.shape)
                                grad_cache, total_loss = gc.build_cache(cls_hiddens) #.squeeze()
                                grad_cache = grad_cache[0]
                                if local_rank > -1:
                                    total_loss = total_loss / dist.get_world_size()

                            elif pretrain:
                                pos_in += [batch["seq_tokens"], batch["seq_labels"]]  # seq_labels contain the -100! #seq_tokens_original #seq_masked
                            else:
                                pos_in += [batch["query_tokens"],batch["doc_pos_tokens"]]
                                neg_in += [batch["query_tokens"],batch["doc_neg_tokens"]]

                            if use_title_body_sep:
                                pos_in.append(batch["title_pos_tokens"])
                                neg_in.append(batch["title_neg_tokens"])

                            if train_qa_spans: # add start positions for qa training (used to anchor end logits on the start ground truth)
                                pos_in.append(batch["pos_qa_start"])

                            #if True:
                            #    pos_in.append(global_i)
                            #    neg_in.append(global_i)

                            #
                            # run model forward
                            #
                            if pretrain and gradient_cache:
                                output_pos = None
                                chunked_inputs = split_tensor_dict(inputs, cache_chunk_size=cache_chunk_size)
                                chunked_labels = split_tensor_dict(batch["seq_labels"], cache_chunk_size=cache_chunk_size)

                                # Compute the full loss with cached gradients
                                for local_chunk_id, chunk in enumerate(chunked_inputs):
                                    device_offset = max(0,local_rank) * per_device_train_batch_size * 2
                                    local_offset = local_chunk_id * cache_chunk_size
                                    chunk_offset = device_offset + local_offset
                                    with torch.cuda.amp.autocast(enabled=use_fp16):
                                        with rnd_states[local_chunk_id]:
                                            pos_in += [chunk, chunked_labels[local_chunk_id]]
                                            ranking_score_pos, mlm_loss, co_loss, surrogate = model.forward(tokens=chunk, labels=chunked_labels[local_chunk_id], grad_cache=grad_cache,
                                                                                       chunk_offset=chunk_offset, use_fp16=use_fp16)

                                    #if gradient_accumulation_steps > -1:
                                    #    raise ValueError

                                    ddp_no_sync = local_rank > -1 and (local_chunk_id + 1 < len(chunked_inputs))
                                    with model.no_sync() if ddp_no_sync else nullcontext():
                                        if is_distributed:
                                            if use_fp16:
                                                (config["mlm_loss_lambda"] * scaler.scale(mlm_loss) + surrogate).sum().backward()
                                            #elif self.use_apex:
                                            #    raise ValueError
                                            #elif self.deepspeed:
                                            #    raise ValueError
                                            else:
                                                (config["mlm_loss_lambda"] * mlm_loss + surrogate).sum().backward()
                                        else:
                                            if use_fp16:
                                                (config["mlm_loss_lambda"] * scaler.scale(mlm_loss) + surrogate).backward()
                                            #elif self.use_apex:
                                            #    raise ValueError
                                            #elif self.deepspeed:
                                            #    raise ValueError
                                            else:
                                                (config["mlm_loss_lambda"] * mlm_loss + surrogate).backward()
                                    total_loss += mlm_loss

                                    # maybe this leads to OOM?
                                    if output_pos is None:
                                        output_pos = ranking_score_pos
                                    else:
                                        output_pos = torch.cat((output_pos, ranking_score_pos), 0)
                                #output_pos = None
                                output_neg = None

                            elif pretrain:
                                #print(use_fp16)
                                output_pos = model.forward(*pos_in, use_fp16=use_fp16)
                            else:
                                #print(use_fp16)
                                #print(pos_in)
                                output_pos = model.forward(*pos_in, use_fp16=use_fp16)
                                output_neg = model.forward(*neg_in, use_fp16=use_fp16)

                            #
                            # untangle output
                            #
                            if train_sparsity:
                                output_pos, sparsity_vecs_pos = output_pos
                                output_neg, sparsity_vecs_neg = output_neg

                            if train_per_term_scores:
                                (output_pos, per_term_output_pos) = output_pos
                                (output_neg, per_term_output_neg) = output_neg

                            if dual_loss and use_in_batch_negatives:
                                output_pos, ranking_score_pos_secondary, query_vecs_pos, doc_vecs_pos = output_pos
                                output_neg, ranking_score_neg_secondary, query_vecs_neg, doc_vecs_neg = output_neg
                            elif dual_loss:
                                output_pos, ranking_score_pos_secondary = output_pos
                                output_neg, ranking_score_neg_secondary = output_neg

                            elif use_in_batch_negatives or use_in_batch_queries or use_intra_batch_queries or use_in_batch_queries_length:
                                output_pos, query_vecs_pos, doc_vecs_pos = output_pos
                                output_neg, query_vecs_neg, doc_vecs_neg = output_neg

                            if train_qa_spans:
                                output_pos,answerability_pos,qa_logits_start_pos,qa_logits_end_pos = output_pos
                                output_neg,answerability_neg,qa_logits_start_neg,qa_logits_end_neg = output_neg

                            if pretrain and not gradient_cache:
                                if config['model'] == 'CoCoColBERTer':
                                    if config['cache_chunk_size']:
                                        ranking_score_pos, mlm_loss, co_loss, surrogate = output_pos
                                    else:
                                        ranking_score_pos, mlm_loss, co_loss = output_pos
                                else:
                                    ranking_score_pos, mlm_loss = output_pos

                                output_pos = ranking_score_pos
                                output_neg = None
                            
                            #if use_list_loss:
                            #    scores = torch.cat([output_pos.unsqueeze(1),output_neg.view(output_pos.shape[0],-1)],dim=-1)
                            #    ranks = scores.sort(descending=True,dim=-1)[1]
                            #    positions = torch.arange(1,1 + ranks.shape[1],device=scores.device)
                            #    mrr = 1 / positions.unsqueeze(0).expand(ranks.shape[0],-1)[ranks == 0].float()
                            #    mrr_avg += mrr.detach().mean()
                            #    output_pos = torch.repeat_interleave(output_pos,output_neg.shape[0] // output_pos.shape[0],dim=0)

                            else:
                                ranking_score_pos = output_pos
                                ranking_score_neg = output_neg


                            #
                            # loss & stats computation
                            #                           
                            if ranking_score_pos.shape[0] != training_batch_size:                              # the last batches might (will) be smaller
                                print("ranking_score_pos has not the same shape as the training_batch_size")
                                label = torch.ones(ranking_score_pos.shape[0],device=ranking_score_pos.device) # but it should only affect the last n batches

                            with torch.cuda.amp.autocast(enabled=use_fp16):
                                #if use_list_loss:
                                #   label=batch["labels"]
                                #   loss = ranking_loss_fn(scores, label)
                                #el
                                if train_pairwise_distillation:
                                   loss = ranking_loss_fn(ranking_score_pos, ranking_score_neg, batch[teacher_pos_key], batch[teacher_neg_key])

                                   #if train_per_term_scores:

                                   #    per_term_output_pos = per_term_output_pos[batch["dyn_teacher_per_term_scores_pos"]>-1000]
                                   #    per_term_output_neg = per_term_output_neg[batch["dyn_teacher_per_term_scores_neg"]>-1000]

                                   #    per_term_labels_pos = batch["dyn_teacher_per_term_scores_pos"][batch["dyn_teacher_per_term_scores_pos"]>-1000]
                                   #    per_term_labels_neg = batch["dyn_teacher_per_term_scores_neg"][batch["dyn_teacher_per_term_scores_neg"]>-1000]

                                   #    #lt1 = (per_term_output_pos.mean(-1,keepdim=True).detach() - per_term_output_pos) - (per_term_labels_pos.mean(-1,keepdim=True) - per_term_labels_pos).detach()
                                   #    #lt2 = (per_term_output_neg.mean(-1,keepdim=True).detach() - per_term_output_neg) - (per_term_labels_neg.mean(-1,keepdim=True) - per_term_labels_neg).detach()

                                   #    lt1 = per_term_output_pos - per_term_labels_pos
                                   #    lt2 = per_term_output_neg - per_term_labels_neg

                                   #    per_term_loss = 0.5 * (torch.mean(torch.pow(lt1, 2)) + torch.mean(torch.pow(lt2, 2)))
                                   #    loss = per_term_loss
                                   #    tensorboard_stats["PairwiseRankScore/PerTermLoss"] += per_term_loss.detach().data


                                       #loss = loss + 0.2 * torch.mean(torch.pow((per_term_output_pos[:,0,:] - per_term_output_pos) - (batch["dyn_teacher_per_term_scores_pos"][:,0,:] - batch["dyn_teacher_per_term_scores_pos"]),2))
                                       #loss = loss + 0.2 * torch.mean(torch.pow((per_term_output_neg[:,0,:] - per_term_output_neg) - (batch["dyn_teacher_per_term_scores_neg"][:,0,:] - batch["dyn_teacher_per_term_scores_neg"]),2))
                                elif pretrain and gradient_cache:
                                    loss = total_loss

                                elif pretrain:
                                    loss = ranking_loss_fn(ranking_score_pos, batch["seq_labels"]) #batch["seq_tokens_original"],
                                    if late_mlm:
                                        loss = loss + config["mlm_loss_lambda"] * mlm_loss
                                    if config["model"] == 'CoCoColBERTer':
                                        loss = loss + config["cocondenser_loss_lambda"] * co_loss + config["mlm_loss_lambda"] * mlm_loss
                                    #if grad_cache is None:
                                    #    co_loss =
                                    #else:
                                    #    loss = loss * (float(hiddens.size(0)) / self.train_args.per_device_train_batch_size)
                                    #    cached_grads = grad_cache[chunk_offset: chunk_offset + co_cls_hiddens.size(0)]
                                    #    surrogate = torch.dot(cached_grads.flatten(), co_cls_hiddens.flatten())
                                    #    return loss, surrogate
                                else:
                                    loss = ranking_loss_fn(ranking_score_pos, ranking_score_neg, label)

                                if pretrain:
                                    tensorboard_stats["PairwiseRankScore/Loss"]         += loss.detach().data
                                    ranking_score_pos_class = torch.argmax(ranking_score_pos, dim=2)
                                    # only eval on the ones which are not -100
                                    tp = torch.sum(torch.eq(ranking_score_pos_class, batch["seq_labels"]["input_ids"])==True).float()
                                    all = torch.sum(torch.eq(batch["seq_labels"]["input_ids"], -100)==False).float()
                                    tensorboard_stats["PairwiseRankScore/Accuracy"] += (tp/all).detach().data
                                else:
                                    tensorboard_stats["PairwiseRankScore/Loss"]         += loss.detach().data
                                    tensorboard_stats["PairwiseRankScore/pos_avg"]      += ranking_score_pos.detach().mean().data
                                    tensorboard_stats["PairwiseRankScore/neg_avg"]      += ranking_score_neg.detach().mean().data
                                    if len(ranking_score_pos.shape) == 1:
                                        tensorboard_stats["PairwiseRankScore/score_diff"]   += (ranking_score_pos - ranking_score_neg).detach().mean().data
                                        tensorboard_stats["PairwiseRankScore/Accuracy"]     += (ranking_score_pos > ranking_score_neg).float().mean().detach().data

                                if dual_loss and train_pairwise_distillation:
                                    loss_secondary = ranking_loss_fn(ranking_score_pos_secondary, ranking_score_neg_secondary, batch[teacher_pos_key], batch[teacher_neg_key])

                                    tensorboard_stats["PairwiseRankScore_Secondary/Loss"]         += loss.detach().data                                
                                    tensorboard_stats["PairwiseRankScore_Secondary/pos_avg"]      += ranking_score_pos.detach().mean().data
                                    tensorboard_stats["PairwiseRankScore_Secondary/neg_avg"]      += ranking_score_neg.detach().mean().data
                                    if len(ranking_score_pos.shape) == 1:
                                        tensorboard_stats["PairwiseRankScore_Secondary/score_diff"]   += (ranking_score_pos - ranking_score_neg).detach().mean().data
                                        tensorboard_stats["PairwiseRankScore_Secondary/Accuracy"]     += (ranking_score_pos > ranking_score_neg).float().mean().detach().data

                                    loss = loss + (loss_secondary * config["secondary_loss_lambda"])
                                #
                                # in-batch negative sampling, we combine it outside forward() to get a multi-gpu sync over the whole batch
                                # -> only select in-batch pairs that are not covered by the ranking loss above
                                #
                                if use_in_batch_negatives:
                                    ib_select  = torch.ones((doc_vecs_neg.shape[0],doc_vecs_neg.shape[0]),device=doc_vecs_neg.device)
                                    ib_select.fill_diagonal_(0)
                                    ib_select = ib_select.bool()

                                    if config["model"] == "ColBERT":
                                        ib_scores_pos_docs = model.forward_inbatch_aggregation(query_vecs_pos, batch["query_tokens"]["attention_mask"], 
                                                                                               doc_vecs_pos, batch["doc_pos_tokens"]["attention_mask"])
                                        ib_scores_neg_docs = model.forward_inbatch_aggregation(query_vecs_pos, batch["query_tokens"]["attention_mask"], 
                                                                                               doc_vecs_neg, batch["doc_neg_tokens"]["attention_mask"])

                                    else: # bert_dot models
                                        ib_scores_pos_docs = torch.mm(query_vecs_pos, doc_vecs_pos.transpose(-2,-1))#[ib_select]
                                        ib_scores_neg_docs = torch.mm(query_vecs_pos, doc_vecs_neg.transpose(-2,-1))#[ib_select]

                                    expanded_pos_scores = ranking_score_pos.unsqueeze(-1).expand(-1,ranking_score_pos.shape[0]-1).reshape(-1)
                                    if use_dynamic_teacher:

                                        if use_inbatch_list_loss:
                                            ib_scores_cat = torch.cat([ib_scores_pos_docs,ib_scores_neg_docs],dim=1)
                                            ib_label_cat = torch.cat([batch["dyn_teacher_scores_pos"],batch["dyn_teacher_scores_neg"]],dim=1)

                                            ib_loss = inbatch_loss_fn(ib_scores_cat,ib_label_cat)
                                        else:
                                            ib_label_pos_only = batch["dyn_teacher_scores_pos"][~ib_select].unsqueeze(-1).expand(-1,ranking_score_pos.shape[0]-1).reshape(-1)
                                            ib_label_pos = batch["dyn_teacher_scores_pos"][ib_select]
                                            ib_label_neg = batch["dyn_teacher_scores_neg"][ib_select]

                                            ib_loss = (inbatch_loss_fn(expanded_pos_scores,ib_scores_pos_docs[ib_select],ib_label_pos_only,ib_label_pos) +\
                                                       inbatch_loss_fn(expanded_pos_scores,ib_scores_neg_docs[ib_select],ib_label_pos_only,ib_label_neg)) * 0.5

                                            tensorboard_stats["In-Batch/Teacher-Accuracy"] += ((ib_label_pos_only > ib_label_pos).float().mean() + (ib_label_pos_only > ib_label_neg).float().mean()).detach().data * 0.5
                                    
                                    #elif train_pairwise_distillation:
                                    #    ib_label_neg = torch.zeros((expanded_pos_scores.shape[0]),device=expanded_pos_scores.device)
                                    #    ib_label_pos = torch.ones((expanded_pos_scores.shape[0]),device=expanded_pos_scores.device) \
                                    #                    * (approximate_margin_factor_inb_negs * (batch[teacher_pos_key] - batch[teacher_neg_key]).mean())
#
                                    #    ib_loss = (inbatch_loss_fn(expanded_pos_scores,ib_scores_pos_docs[ib_select],ib_label_pos.detach(),ib_label_neg) +\
                                    #               inbatch_loss_fn(expanded_pos_scores,ib_scores_neg_docs[ib_select],ib_label_pos.detach(),ib_label_neg)) * 0.5

                                    else:
                                        ib_label = torch.ones((expanded_pos_scores.shape[0]),device=expanded_pos_scores.device)

                                        ib_loss = (inbatch_loss_fn(expanded_pos_scores,ib_scores_pos_docs[ib_select],ib_label) +\
                                                   inbatch_loss_fn(expanded_pos_scores,ib_scores_neg_docs[ib_select],ib_label)) * 0.5

                                    loss = loss * config["in_batch_main_pair_lambda"] + ib_loss * config["in_batch_neg_lambda"]
                                    #loss, weighted_losses = merge_loss([loss,ib_loss],log_var_mtl)

                                    tensorboard_stats["In-Batch/Loss"]         += ib_loss.detach().data
                                    tensorboard_stats["In-Batch/Accuracy"]     += ((expanded_pos_scores > ib_scores_pos_docs[ib_select]).float().mean() + (expanded_pos_scores > ib_scores_neg_docs[ib_select]).float().mean()).detach().data * 0.5
                                    tensorboard_stats["In-Batch/score_diff"]   += ((expanded_pos_scores - ib_scores_pos_docs[ib_select]).mean() + (expanded_pos_scores - ib_scores_neg_docs[ib_select]).mean()).detach().data * 0.5

                                if use_in_batch_queries:
                                    #print('use_in_batch_queries_loss')
                                    ib_select = torch.ones((query_vecs_pos.shape[0], query_vecs_pos.shape[0]),
                                                           device=query_vecs_pos.device)
                                    ib_select.fill_diagonal_(0)
                                    ib_select = ib_select.bool()

                                    if config["in_batch_which"] == 'negatives':
                                        print('in batch use negatives')
                                        ib_query_distances = torch.mm(doc_vecs_neg,
                                                                      doc_vecs_neg.transpose(-2, -1))  # [ib_select]
                                    elif config["in_batch_which"] == 'positives':
                                        print('in batch use positives')
                                        ib_query_distances = torch.mm(doc_vecs_pos,
                                                                      doc_vecs_pos.transpose(-2, -1))  # [ib_select]
                                    else:
                                        print('in batch use queries')
                                        ib_query_distances = torch.mm(query_vecs_pos,
                                                                      query_vecs_pos.transpose(-2, -1))  # [ib_select]

                                    ib_loss = inbatch_loss_query(ib_query_distances[ib_select])

                                    loss = loss * config["in_batch_main_pair_query_lambda"] + ib_loss * config[
                                        "in_batch_query_lambda"]
                                    # loss, weighted_losses = merge_loss([loss,ib_loss],log_var_mtl)

                                    tensorboard_stats["In-Batch/QLoss"] += ib_loss.detach().data
                                    tensorboard_stats["In-Batch/QDistance"] += (ib_query_distances[ib_select]).float().mean()


                                # use intra-batch query distance
                                #
                                if use_intra_batch_queries:
                                    if query_vecs_pos_prev_batch is not None:
                                        #print('use_intra_batch_queries_loss')
                                        # only starts at the second one
                                        #query_vecs_pos_prev_batch = torch.autograd.Variable(query_vecs_pos_prev_batch)
                                        if config["intra_batch_which"] == 'negatives':
                                            print('intra batch use negatives')
                                            ib_query_distances = torch.mm(query_vecs_pos_prev_batch,
                                                                          doc_vecs_neg.transpose(-2, -1))  # [ib_select]
                                        elif config["intra_batch_which"] == "positives":
                                            print('intra batch use positives')
                                            ib_query_distances = torch.mm(query_vecs_pos_prev_batch,
                                                                          doc_vecs_pos.transpose(-2, -1))  # [ib_select]
                                        else:
                                            print('intra batch use queries')
                                            ib_query_distances = torch.mm(query_vecs_pos_prev_batch,
                                                                          query_vecs_pos.transpose(-2, -1))  # [ib_select]

                                        ib_loss = inbatch_loss_query(ib_query_distances)

                                        loss = loss * config["intra_batch_main_pair_query_lambda"] + ib_loss * config[
                                            "intra_batch_query_lambda"]
                                        # loss, weighted_losses = merge_loss([loss,ib_loss],log_var_mtl)
                                        tensorboard_stats["In-Batch/IntraQLoss"] += ib_loss.detach().data
                                        tensorboard_stats["In-Batch/IntraQDistance"] += (ib_query_distances).float().mean()

                                    if config["intra_batch_which"] == 'negatives':
                                        query_vecs_pos_prev_batch = doc_vecs_neg.detach()
                                    elif config["intra_batch_which"] == "positives":
                                        query_vecs_pos_prev_batch = doc_vecs_pos.detach()
                                    else:
                                        query_vecs_pos_prev_batch = query_vecs_pos.detach()

                                # use in batch query distance
                                #
                                if use_in_batch_queries_length:
                                    #print('use_in_batch_queries_loss')
                                    ib_select = torch.zeros((query_vecs_pos.shape[0], query_vecs_pos.shape[0]),
                                                           device=query_vecs_pos.device)
                                    ib_select.fill_diagonal_(1)
                                    ib_select = ib_select.bool()

                                    ib_query_lengths = torch.mm(query_vecs_pos,
                                                                  query_vecs_pos.transpose(-2, -1))
                                    #ib_pos_lengths = torch.mm(doc_vecs_pos,
                                    #                              doc_vecs_pos.transpose(-2, -1))

                                    ib_loss_query = inbatch_loss_query_length(ib_query_lengths[ib_select])
                                    #ib_loss_pos = inbatch_loss_query_length(ib_pos_lengths[ib_select])

                                    loss = loss * config["in_batch_main_pair_query_length_lambda"] \
                                           + ib_loss_query * config["in_batch_query_length_lambda"]
                                           #+ ib_loss_pos * config["in_batch_pos_length_lambda"]
                                    # loss, weighted_losses = merge_loss([loss,ib_loss],log_var_mtl)

                                    tensorboard_stats["In-Batch/QLengthLoss"] += ib_loss.detach().data
                                    tensorboard_stats["In-Batch/QLength"] += (ib_query_lengths[ib_select]).float().mean()


                                if train_qa_spans:


                                    qa_loss,answ_loss = qa_loss_fn(qa_logits_start_pos,qa_logits_end_pos,batch["pos_qa_start"],batch["pos_qa_end"],answerability_pos,batch["pos_qa_hasAnswer"])
                                    #qa_loss_neg = qa_loss_fn(qa_logits_start_neg,qa_logits_end_neg,batch["neg_qa_start"],batch["neg_qa_end"]) #,answerability_pos,batch["pos_qa_hasAnswer"])
                                    qa_loss_neg,answ_loss_neg = qa_loss_fn(None,None,None,None,answerability_neg,torch.zeros(answerability_neg.shape[0],device=answerability_neg.device,dtype=torch.int64))

                                    loss,weighted_losses = merge_loss([loss,qa_loss,answ_loss+0.1*answ_loss_neg]) # * config["qa_loss_lambda"]
                                    tensorboard_stats["QA/Loss_weighted_ranking"] += weighted_losses[0]
                                    tensorboard_stats["QA/Loss_weighted_qa"]      += weighted_losses[1]
                                    tensorboard_stats["QA/Loss"]            += qa_loss.mean().detach().data
                                    tensorboard_stats["QA/Start_accuracy"]  += (torch.max(qa_logits_start_pos,dim=-1).indices.unsqueeze(-1) == batch["pos_qa_start"]).any(-1).float().mean().detach().data
                                    tensorboard_stats["QA/End_accuracy"]    += ((torch.max(qa_logits_end_pos,dim=-1).indices.unsqueeze(-1) == batch["pos_qa_end"].unsqueeze(1)).any(-1)[batch["pos_qa_end"] > -1]).float().mean().detach().data

                                    #tensorboard_stats["QA/Answerability_pos"]           += answerability_pos.mean().detach().data
                                    tensorboard_stats["QA/Answerability_pos_accuracy"]  += (torch.max(answerability_pos,dim=-1).indices == batch["pos_qa_hasAnswer"]).float().mean().detach().data
                                    tensorboard_stats["QA/Answerability_neg_accuracy"]  += (torch.max(answerability_neg,dim=-1).indices == 0).float().mean().detach().data
                                    #tensorboard_stats["QA/Answerability_neg"]           += answerability_neg.mean().detach().data

                                if train_sparsity:
                                    tensorboard_stats["Sparsity/sparsity_avg_pos"] += (sparsity_vecs_pos[0].detach().data <= 0).float().mean() 
                                    tensorboard_stats["Sparsity/sparsity_avg_neg"] += (sparsity_vecs_neg[0].detach().data <= 0).float().mean()

                                    sparsity_loss = sparsity_vecs_pos[0].mean() + sparsity_vecs_neg[0].mean()

                                    #sparsity_loss2 = torch.zeros(1).cuda(cuda_device)
                                    #for tens in itertools.chain.from_iterable(sparsity_vecs_pos + sparsity_vecs_neg):
                                    #    sparsity_loss2 = sparsity_loss2 + tens.mean()

                                    loss = loss + (config["sparsity_loss_lambda_factor"] * sparsity_loss)

                                    tensorboard_stats["Sparsity/loss"] += sparsity_loss.detach().data

                            if not gradient_cache:
                                if use_fp16:
                                    scaler.scale(loss).backward()
                                    if gradient_accumulation:
                                        if (1+i)%gradient_accumulation_steps == 0:
                                            scaler.step(optimizer)
                                            if use_embedding_optimizer:
                                                scaler.step(embedding_optimizer)
                                                embedding_optimizer.zero_grad()
                                            scaler.update()
                                            optimizer.zero_grad()
                                    #elif gradient_cache:
                                    else:
                                        scaler.unscale_(optimizer)
                                        norm_temp = torch.nn.utils.clip_grad_norm_(model.parameters(), 10_000,error_if_nonfinite=False)
                                        if not torch.isnan(norm_temp):
                                            tensorboard_stats["Gradients/GradNorm"] += norm_temp.data.detach()

                                        scaler.step(optimizer)
                                        if use_embedding_optimizer:
                                            scaler.step(embedding_optimizer)
                                            embedding_optimizer.zero_grad()
                                        scaler.update()
                                        optimizer.zero_grad()
                                else:
                                    loss.backward()
                                    if gradient_accumulation:
                                        if (1+i)%gradient_accumulation_steps == 0:
                                            optimizer.step()
                                            if use_embedding_optimizer:
                                                embedding_optimizer.step()
                                                embedding_optimizer.zero_grad()
                                            optimizer.zero_grad()
                                    #elif gradient_cache:

                                    else:
                                        norm_temp = torch.nn.utils.clip_grad_norm_(model.parameters(), 10_000,error_if_nonfinite=False)
                                        if not torch.isnan(norm_temp):
                                            tensorboard_stats["Gradients/GradNorm"] += norm_temp.data.detach()
                                        optimizer.step()
                                        if use_embedding_optimizer:
                                            embedding_optimizer.step()
                                            embedding_optimizer.zero_grad()
                                        optimizer.zero_grad()

                            # set the label back to a cached version (for next iterations)
                            if output_pos.shape[0] != training_batch_size:
                                label = torch.ones(training_batch_size, device=output_pos.device)

                            #
                            # reporting 
                            #
                            training_step_write = config.get('training_step_write', 100)
                            if i > 0 and i % training_step_write == 0:
                                lr_scheduler.step()
                                if embedding_optimizer is not None:
                                    lr_scheduler2.step()

                                # append loss to loss file
                                with open(loss_file_path, "a") as loss_file:
                                    loss_file.write(str(epoch) + "," +str(i) + "," + str(tensorboard_stats["PairwiseRankScore/Loss"].item()/100) +"\n")

                                if train_sparsity:
                                    if config["sparsity_reanimate"] == True and (tensorboard_stats["Sparsity/sparsity_avg"]/100 > 0.9).any():
                                        if is_distributed:
                                            model.module.reanimate(config["sparsity_reanimate_bias"],(tensorboard_stats["Sparsity/sparsity_avg"]/100 > 0.9))
                                        else:
                                            model.reanimate(config["sparsity_reanimate_bias"],(tensorboard_stats["Sparsity/sparsity_avg"]/100 > 0.9))
                                if use_fp16:
                                    tensorboard_stats["Gradients/Scaler"] += scaler.get_scale()

                                for ts_name,ts_value in tensorboard_stats.items():
                                    tb_writer.add_scalar(ts_name, ts_value.item()/100, global_i)
                                    ts_value*=0
                                    if torch.isnan(ts_value):
                                        tensorboard_stats[ts_name] = torch.zeros(1).cuda(cuda_device)

                                status.update("[bold magenta]           Progress ... Batch No.: "+str(i), refresh=True)

                        except RuntimeError as r:
                            if r.args[0].startswith("CUDA out of memory"): # lol yeah that is python for you 
                                if i - lastRetryI < 2:
                                    raise r

                                del loss,output_neg,output_pos,batch
                                gc.collect()
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                                time.sleep(5)
                                lastRetryI=i
                                logger.warning("["+str(i)+"] Caught CUDA OOM: " + r.args[0]+", now cached:"+str(torch.cuda.memory_reserved()))
                                print("["+str(i)+"] Caught CUDA OOM: Retrying next batch ... ")
                            else:
                                raise r
                            
                        #
                        # vars we need for the training loop 
                        # -------------------------------
                        #
                        if do_validate_every_n_batches:
                            if i > 0 and i % validate_every_n_batches == 0 or (train_subset and epoch % validate_every_n_epochs == 0 and i==1):
                                status.update("[bold magenta]           Running validation, after batch no.: "+str(i), refresh=True)

                                if len(per_cluster_idx_diff) > 0:
                                    with open(os.path.join(run_folder, "cluster-loss-diff.tsv"), "w") as loss_file:
                                        for c_idx,c_loss in per_cluster_idx_diff.items():
                                            loss_file.write(str(c_idx)+"\t"+str(sum(c_loss)/len(c_loss))+"\t"+" ".join([str(a) for a in c_loss])+"\n")

                                perf_monitor.log_value("train_gpu_mem",torch.cuda.memory_allocated()/float(1e9))
                                perf_monitor.log_value("train_gpu_mem_max",torch.cuda.max_memory_allocated()/float(1e9))
                                perf_monitor.log_value("train_gpu_cache",torch.cuda.memory_reserved()/float(1e9))
                                perf_monitor.log_value("train_gpu_cache_max",torch.cuda.max_memory_reserved()/float(1e9))
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                                torch.cuda.reset_peak_memory_stats()

                                perf_monitor.stop_block("train",(i - perf_start_inst) * training_batch_size)
                                perf_start_inst = i

                                if pretrain:
                                    best_metric = {}
                                    best_metric["metrics"] = {}
                                    best_metric["cs@n"] = "-"
                                    best_metric["epoch"] = epoch
                                    best_metric["batch_number"] = i
                                    best_metric["metrics"][config["validation_metric"]] = (tp/all).cpu().detach().data.numpy()
                                else:
                                    best_metric, _, validated_count,qa_validation_results = validate_model("cont",model, config,config["validation_cont"], run_folder, logger, cuda_device, epoch, i,
                                                                                                           best_metric_info,validation_cont_candidate_set,use_cache=config["validation_cont_use_cache"],
                                                                                                           output_secondary_output=False,is_distributed=is_distributed,status=status)
                                    for k,v in qa_validation_results.items():
                                        tb_writer.add_scalar('Validation'+k, v, global_i)

                                    tb_writer.add_scalar('Validation/nDCG@3', best_metric["metrics"]["nDCG@3"], global_i)
                                    tb_writer.add_scalar('Validation/nDCG@10', best_metric["metrics"]["nDCG@10"], global_i)
                                    tb_writer.add_scalar('Validation/MRR@10', best_metric["metrics"]["MRR@10"], global_i)
                                if best_metric["cs@n"] != "-":
                                    tb_writer.add_scalar('Validation/Depth', best_metric["cs@n"], global_i)

                                if best_metric["metrics"][config["validation_metric"]] > best_metric_info["metrics"][config["validation_metric"]]:
                                    best_metric_info = best_metric
                                    save_best_info(best_metric_info_file,best_metric_info)
                                    if is_distributed:
                                        torch.save(model.module.state_dict(), best_model_store_path)
                                    else:
                                        if store_n_best_checkpoints > 1:
                                            # best_3 > best_4
                                            # best_2 > best_3 
                                            # best_model path > best_2
                                            for n_check in list(range(2, store_n_best_checkpoints))[::-1]:
                                                path_a = os.path.join(run_folder,str(n_check)+"-best-model.pytorch-state-dict")
                                                path_b = os.path.join(run_folder,str(n_check + 1)+"-best-model.pytorch-state-dict")
                                                if os.path.exists(path_a):
                                                    os.replace(path_a,path_b)
                                            if os.path.exists(best_model_store_path):
                                                os.replace(best_model_store_path,os.path.join(run_folder,"2-best-model.pytorch-state-dict"))
                                        else:
                                            torch.save(model.state_dict(), best_model_store_path)

                                    logger.info(str(i)+"Saved new best weights with: "+config["validation_metric"]+": " + str(best_metric["metrics"][config["validation_metric"]]))


                                if is_distributed:
                                    logger.info(str(i)+"-"+model.module.get_param_stats())
                                else:
                                    logger.info(str(i)+"-"+model.get_param_stats())

                                model.train()

                                tb_writer.add_scalar('Learning Rate/Group 0', optimizer.param_groups[0]["lr"], global_i)
                                tb_writer.add_scalar('Learning Rate/Group 1', optimizer.param_groups[1]["lr"], global_i)

                                if early_stopper.step(best_metric["metrics"][config["validation_metric"]]) and global_i > min_steps_training:
                                    logger.info("early stopping epoch %i batch count %i",epoch,i)
                                    print(run_folder)
                                    break

                                perf_monitor.log_value("eval_gpu_mem",torch.cuda.memory_allocated()/float(1e9))
                                perf_monitor.log_value("eval_gpu_mem_max",torch.cuda.max_memory_allocated()/float(1e9))
                                perf_monitor.log_value("eval_gpu_cache",torch.cuda.memory_reserved()/float(1e9))
                                perf_monitor.log_value("eval_gpu_cache_max",torch.cuda.max_memory_reserved()/float(1e9))
                                torch.cuda.empty_cache()
                                torch.cuda.reset_peak_memory_stats()

                                perf_monitor.start_block("train")

                                if global_i == validate_every_n_batches * 2: # Print after the second to allow for cached validation results 
                                    #console.log("\nPerformance report:              \n")
                                    status.update("", refresh=True)
                                    perf_monitor.print_summary(console)
                                    perf_monitor.save_summary(os.path.join(run_folder,"efficiency-metrics.json"))

                        if pretrain and i > 0 and i % 100000 == 0:
                            torch.save(model.state_dict(), os.path.join(run_folder, "step_" + str(
                                i) + "-best-model.pytorch-state-dict"))

                        i+=1
                        global_i+=1

                if epoch in [15, 35, 60, 85, 110, 135, 160]:  # minus 15
                    shutil.copyfile(best_model_store_path, os.path.join(run_folder, "epoch_" + str(
                        epoch + 15) + "-best-model.pytorch-state-dict"))

                perf_monitor.stop_block("train",i - perf_start_inst)


            if is_distributed:
                torch.save(model.module.state_dict(), os.path.join(run_folder,"final-model.pytorch-state-dict"))
            else:
                torch.save(model.state_dict(), os.path.join(run_folder,"final-model.pytorch-state-dict"))

        #
        # evaluate the end validation & test & leaderboard sets with the best model checkpoint
        #
        print("Done with training! Reloading best checkpoint ...")
        #print("Mem allocated:",torch.cuda.memory_allocated())

        if is_distributed:
            model_cpu = model.module.cpu() # we need this strange back and forth copy for models > 1/2 gpu memory, because load_state copies the state dict temporarily
        else:
            model_cpu = model.cpu() # we need this strange back and forth copy for models > 1/2 gpu memory, because load_state copies the state dict temporarily

        del model, optimizer, all_params, we_params, lr_scheduler
        if from_scratch:
            del loss
        if config["train_embedding"]:
            del embedding_optimizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(10) # just in case the gpu has not cleaned up the memory
        torch.cuda.reset_peak_memory_stats()
        model_cpu.load_state_dict(torch.load(best_model_store_path,map_location="cpu"),strict=False)
        model = model_cpu.cuda(cuda_device)
        if is_distributed:
            model = nn.DataParallel(model)

        print("Model reloaded ! memory allocation:",torch.cuda.memory_allocated())

        best_validation_end_metrics={}
        if "validation_end" in config:
            for validation_end_name,validation_end_config in config["validation_end"].items():
                print("Evaluating validation_end."+validation_end_name)
                
                validation_end_candidate_set = None
                if "candidate_set_path" in validation_end_config:
                    validation_end_candidate_set = parse_candidate_set(validation_end_config["candidate_set_path"],validation_end_config["candidate_set_from_to"][1])
                best_metric, _, validated_count,_ = validate_model(validation_end_name,model, config,validation_end_config,
                                                                 run_folder, logger, cuda_device, 
                                                                 candidate_set=validation_end_candidate_set,
                                                                 output_secondary_output=validation_end_config["save_secondary_output"],is_distributed=is_distributed)
                save_best_info(os.path.join(run_folder, "val-"+validation_end_name+"-info.csv"),best_metric)
                best_validation_end_metrics[validation_end_name] = best_metric

        if "test" in config:
            for test_name,test_config in config["test"].items():
                print("Evaluating test."+test_name)
                cs_at_n_test=None
                test_candidate_set = None
                if "candidate_set_path" in test_config:
                    cs_at_n_test = best_validation_end_metrics[test_name]["cs@n"]
                    test_candidate_set = parse_candidate_set(test_config["candidate_set_path"],test_config["candidate_set_max"])
                test_result = test_model(model, config,test_config, run_folder, logger, cuda_device,"test_"+test_name,
                                         test_candidate_set, cs_at_n_test,output_secondary_output=test_config["save_secondary_output"],is_distributed=is_distributed)

        if "leaderboard" in config:
            for test_name,test_config in config["leaderboard"].items():
                print("Evaluating leaderboard."+test_name)
                test_model(model, config, test_config, run_folder, logger, cuda_device, "leaderboard"+test_name,output_secondary_output=test_config["save_secondary_output"],is_distributed=is_distributed)

        perf_monitor.log_value("eval_gpu_mem",torch.cuda.memory_allocated()/float(1e9))
        perf_monitor.log_value("eval_gpu_mem_max",torch.cuda.max_memory_allocated()/float(1e9))
        perf_monitor.log_value("eval_gpu_cache",torch.cuda.memory_reserved()/float(1e9))
        perf_monitor.log_value("eval_gpu_cache_max",torch.cuda.max_memory_reserved()/float(1e9))
        torch.cuda.reset_peak_memory_stats()

        perf_monitor.save_summary(os.path.join(run_folder,"efficiency-metrics.json"))

        if config.get("run_dense_retrieval_eval",False):
            print("Starting dense_retrieval")
            import sys ,subprocess
            if config["model"] == "ColBERTer" or config["model"] == "CoColBERTer" or config["model"] == "CoCoColBERTer":
                subprocess.Popen(["python", "matchmaker/colberter_retrieval.py","encode+index+search","--config",config["colberter_retrieval_config"]
                                 ,"--run-name",args.run_name,"--config-overwrites", "trained_model: "+run_folder])
                print(run_folder)
            else:
                subprocess.Popen(["python", "matchmaker/dense_retrieval.py","encode+index+search","--config",config["dense_retrieval_config"]
                                 ,"--run-name",args.run_name,"--config-overwrites", "trained_model: "+run_folder])
                print(run_folder)
        print(run_folder)


    except Exception as e:
        logger.info('-' * 20)
        logger.exception('[train] Got exception: ')
        logger.info('Exiting from training early')
        console.log("[red]Exception! ",str(e))
        console.print_exception()

        print(run_folder)
        exit(1)
