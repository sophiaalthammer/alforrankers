import torch
import numpy
import random

from allennlp.data.samplers import BucketBatchSampler, MaxTokensBatchSampler
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.data_loaders import MultiProcessDataLoader

from transformers import T5Tokenizer

from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer


from matchmaker.dataloaders.concatenated_reranking_loader import *
from matchmaker.dataloaders.concatenated_training_loader import *

from matchmaker.dataloaders.independent_reranking_loader import *
from matchmaker.dataloaders.independent_training_loader import *

from matchmaker.dataloaders.id_sequence_loader import *
from matchmaker.dataloaders.mlm_masked_sequence_loader import *
from matchmaker.dataloaders.query_generation_inference_loader import ConditionalQueryGenerationInferenceReader
from matchmaker.dataloaders.tas_balanced_training_loader import *
from matchmaker.dataloaders.pseudo_label_training_loader import PseudoLabelDatasetLoader, PseudoLabelTextDatasetLoader
from matchmaker.dataloaders.triple_id_training_loader import TripleIdDatasetLoader
from transformers import AutoTokenizer

from matchmaker.dataloaders.bling_fire_tokenizer import BlingFireTokenizer
from matchmaker.dataloaders.transformer_tokenizer import FastTransformerTokenizer
from matchmaker.modules.bert_embedding_token_embedder import PretrainedBertIndexerNoSpecialTokens


from typing import Dict, Tuple, List
#from tokenizers import ByteLevelBPETokenizer,CharBPETokenizer
#from matchmaker.dataloaders.transformer_tokenizer import CustomTransformerTokenizer,CustomTransformerIndexer

import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system") # VERY MUCH needed for linux !! makes everything faster, but tends to break stuff


def allennlp_single_sequence_loader(model_config, run_config, _input_file, sequence_type, force_exact_batch_size=False):
    '''
    Load examples from a .tsv file in the single sequence format: id<tab>text

    (Using allennlp's v2 multiprocess loader)
    '''
    if model_config.get("model_input_type", "") == "mlm":
        sequence_type == "single_mlm"

    if sequence_type == "query":
        max_length = run_config.get("overwrite_max_query_length", model_config["max_query_length"])
        min_length = model_config.get("min_query_length",-1)
        batch_size = run_config["query_batch_size"]
        split_document=False
        split_document_window_size=-1
    if sequence_type == "single_mlm":
        max_length = run_config.get("overwrite_max_doc_length", model_config["max_doc_length"])
        min_length = model_config.get("min_doc_length", -1)
        batch_size = run_config.get("collection_batch_size", run_config["batch_size_train"])
        make_multiple_of=run_config.get("make_multiple_of",8)
        mask_probability=run_config.get("mask_probability",0.1)
        mlm_mask_replace_probability=run_config.get("mlm_mask_replace_probability",0.5)
        mlm_mask_random_probability=run_config.get("mlm_mask_random_probability",0.5)
    else:  # doc
        max_length = run_config.get("overwrite_max_doc_length", model_config["max_doc_length"])
        min_length = model_config.get("min_doc_length",-1)
        batch_size = run_config["collection_batch_size"]
        split_document=run_config.get("split_document",False)
        split_document_window_size=run_config.get("split_document_window_size",-1)

    _tokenizer, _token_indexers, _vocab = _get_indexer(model_config, max_length)

    #if model_config.get("model_input_type", "") == "mlm":
    #    reader = MLMMaskedSequenceDatasetReader(tokenizer=_tokenizer, token_indexers=_token_indexers,
    #                                            max_doc_length=max_length, min_doc_length=min_length,
    #                                            mask_probability=mask_probability,
    #                                            mlm_mask_replace_probability=mlm_mask_replace_probability,
    #                                            mlm_mask_random_probability=mlm_mask_random_probability,
    #                                            make_multiple_of=make_multiple_of)

    reader = IdSequenceDatasetReader(tokenizer=_tokenizer, token_indexers=_token_indexers,
                                     split_document=split_document,split_document_window_size=split_document_window_size,
                                     max_seq_length=max_length, min_seq_length=min_length, sequence_type=sequence_type)

    if force_exact_batch_size:
        loader = MultiProcessDataLoader(reader, data_path=_input_file, num_workers=run_config["dataloader_num_workers"],
                                        max_instances_in_memory=int(batch_size)*25, quiet=True, start_method="fork" if "fork" in mp.get_all_start_methods() else "spawn",
                                        batch_size=int(batch_size))
    else:
        loader = MultiProcessDataLoader(reader, data_path=_input_file, num_workers=run_config["dataloader_num_workers"],
                                        max_instances_in_memory=int(batch_size)*25, quiet=True, start_method="fork" if "fork" in mp.get_all_start_methods() else "spawn",
                                        batch_sampler=MaxTokensBatchSampler(max_tokens=int(batch_size)*max_length, sorting_keys=["seq_tokens"], padding_noise=0))

    loader.index_with(_vocab)
    return loader


def allennlp_triple_training_loader(model_config, run_config, _input_file,add_text_to_batch=False):
    '''
    Load training examples (either in the re-ranking text file format or a dynamic loader)

    (Using allennlp's v2 multiprocess loader)
    '''
    _tokenizer, _token_indexers, _vocab = _get_indexer(model_config, max(run_config["max_doc_length"], run_config["max_query_length"]))

    if run_config.get("dynamic_sampler", False) == False:

        if model_config.get("model_input_type", "") == "concatenated" or model_config["token_embedder_type"] == "bert_cat":

            reader = ConcatenatedTrainingDatasetReader(tokenizer=_tokenizer, token_indexers=_token_indexers,
                                             max_doc_length=run_config["max_doc_length"], max_query_length=run_config["max_query_length"],
                                             min_doc_length=run_config["min_doc_length"], min_query_length=run_config["min_query_length"],
                                             data_augment=run_config["train_data_augment"], train_pairwise_distillation=run_config["train_pairwise_distillation"], 
                                             train_qa_spans=run_config["train_qa_spans"],add_text_to_batch=add_text_to_batch)
        else:
            reader = IndependentTrainingDatasetReader(tokenizer=_tokenizer, token_indexers=_token_indexers,
                                           max_doc_length=run_config["max_doc_length"], max_query_length=run_config["max_query_length"],
                                           min_doc_length=run_config["min_doc_length"], min_query_length=run_config["min_query_length"],
                                           data_augment=run_config["train_data_augment"], train_pairwise_distillation=run_config["train_pairwise_distillation"],
                                           query_augment_mask_number=run_config["query_augment_mask_number"], train_qa_spans=run_config["train_qa_spans"],add_text_to_batch=add_text_to_batch)

        loader = MultiProcessDataLoader(reader, data_path=_input_file, num_workers=run_config["dataloader_num_workers"],
                                        max_instances_in_memory=int(run_config["batch_size_train"])*25, quiet=True, start_method="fork" if "fork" in mp.get_all_start_methods() else "spawn",
                                        batch_size=run_config["batch_size_train"])
        loader.index_with(_vocab)

    else:
        #if run_config["dynamic_sampler_type"] == "list":
        #    loader = IrDynamicTripleDatasetLoader(query_file=run_config["dynamic_query_file"], collection_file=run_config["dynamic_collection_file"],
        #                                          qrels_file=run_config["dynamic_qrels_file"], candidate_file=run_config["dynamic_candidate_file"],
        #                                          batch_size=int(run_config["batch_size_train"]), queries_per_batch=run_config["dynamic_queries_per_batch"], tokenizer=_tokenizer, token_indexers=_token_indexers,
        #                                          max_doc_length=run_config["max_doc_length"], max_query_length=run_config["max_query_length"],
        #                                          min_doc_length=run_config["min_doc_length"], min_query_length=run_config["min_query_length"],
        #                                          data_augment=run_config["train_data_augment"], vocab=_vocab)

        if run_config["dynamic_sampler_type"] == "tas_balanced":
            loader = TASBalancedDatasetLoader(query_file=run_config["dynamic_query_file"], collection_file=run_config["dynamic_collection_file"],
                                              pairs_with_teacher_scores=run_config["dynamic_pairs_with_teacher_scores"], query_cluster_file=run_config["dynamic_query_cluster_file"],
                                              batch_size=int(run_config["batch_size_train"]), clusters_per_batch=run_config["dynamic_clusters_per_batch"], tokenizer=_tokenizer,
                                              max_doc_length=run_config["max_doc_length"], max_query_length=run_config["max_query_length"],
                                              pair_balancing_strategy=run_config["tas_balanced_pair_strategy"],random_seed =run_config["random_seed"])
        elif run_config["dynamic_sampler_type"] == "pseudo_label":
            loader = PseudoLabelDatasetLoader(query_file=run_config["dynamic_query_file"], collection_file=run_config["dynamic_collection_file"],
                                              rankings_with_teacher_scores=run_config["dynamic_rankings_with_teacher_scores"],
                                              selection_type=run_config["pseudo_label_selection_type"],min_pos_score=run_config["pseudo_label_min_pos_score"],
                                              max_diff_to_be_pos=run_config["pseudo_label_max_diff_to_be_pos"],min_diff_to_neg=run_config["pseudo_label_min_diff_to_neg"],
                                              batch_size=int(run_config["batch_size_train"]), tokenizer=_tokenizer,
                                              max_doc_length=run_config["max_doc_length"], max_query_length=run_config["max_query_length"],
                                              random_seed =run_config["random_seed"],concatenate_sequences = model_config.get("model_input_type", "") == "concatenated")
        elif run_config["dynamic_sampler_type"] == "pseudo_labeltext":
            loader = PseudoLabelTextDatasetLoader(rankings_with_teacher_scores=run_config["dynamic_rankings_with_teacher_scores"],
                                              batch_size=int(run_config["batch_size_train"]), tokenizer=_tokenizer,
                                              max_doc_length=run_config["max_doc_length"], max_query_length=run_config["max_query_length"],
                                              random_seed =run_config["random_seed"],concatenate_sequences = model_config.get("model_input_type", "") == "concatenated")
        elif run_config["dynamic_sampler_type"] == "triple_ids":
            loader = TripleIdDatasetLoader(query_file=run_config["dynamic_query_file"], collection_file=run_config["dynamic_collection_file"],
                                              triples_with_teacher_scores=run_config["dynamic_triples_with_teacher_scores"], 
                                              batch_size=int(run_config["batch_size_train"]), tokenizer=_tokenizer,
                                              max_doc_length=run_config["max_doc_length"], max_query_length=run_config["max_query_length"],
                                              random_seed =run_config["random_seed"],concatenate_sequences = model_config.get("model_input_type", "") == "concatenated")

        elif run_config["dynamic_sampler_type"] == "mlm_pretrain":
            loader = MLMDatasetLoader(collection_file=run_config["train_tsv"],
                                        batch_size=int(run_config["batch_size_train"]), tokenizer=_tokenizer,
                                        max_doc_length=run_config["max_doc_length"],
                                        random_seed=run_config["random_seed"],
                                        min_doc_length=-1,
                                        mlm_mask_whole_words=True,
                                        mask_probability=run_config["mask_probability"],
                                        mlm_mask_replace_probability=run_config["mlm_mask_replace_probability"],
                                        mlm_mask_random_probability=run_config["mlm_mask_random_probability"],
                                        whole_word_masking=run_config["whole_word_masking"],
                                        random_spans=run_config["random_spans"],
                                        tasb=run_config["tasb"],
                                        tasb_cluster_file=run_config["tasb_cluster_file"],
                                        tasb_weight=run_config["tasb_weight"],
                                        grad_acc=run_config["gradient_accumulation_steps"],
                                        cached_chunk_size=int(run_config["batch_size_train"])/int(run_config["cache_chunk_size"]))

        else:
            raise ConfigurationError("dynamic sampler type not supported")

    return loader


def allennlp_reranking_inference_loader(model_config, run_config, _input_file):
    '''
    Load examples from a .tsv file in the reranking candidate file format: q_id<tab>d_id<tab>q_text<tab>d_text

    (Using allennlp's v2 multiprocess loader)
    '''

    _tokenizer, _token_indexers, _vocab = _get_indexer(model_config, max(run_config["max_doc_length"], run_config["max_query_length"]))

    if model_config.get("model_input_type", "") == "concatenated" or model_config["token_embedder_type"] == "bert_cat":

        reader = ConcatenatedReRankingDatasetReader(tokenizer=_tokenizer, token_indexers=_token_indexers,
                                                    max_doc_length=run_config["max_doc_length"], max_query_length=run_config["max_query_length"],
                                                    min_doc_length=run_config["min_doc_length"], min_query_length=run_config["min_query_length"],
                                                    train_qa_spans=run_config["train_qa_spans"])
    else:

        reader = IndependentReRankingDatasetReader(tokenizer=_tokenizer, token_indexers=_token_indexers,
                                                   max_doc_length=run_config["max_doc_length"], max_query_length=run_config["max_query_length"],
                                                   min_doc_length=run_config.get("min_doc_length",-1), min_query_length=run_config.get("min_query_length",-1),
                                                   query_augment_mask_number=run_config.get("query_augment_mask_number",-1), train_qa_spans=run_config.get("train_qa_spans",False))

    loader = MultiProcessDataLoader(reader, data_path=_input_file, num_workers=run_config["dataloader_num_workers"],
                                    max_instances_in_memory=int(run_config["batch_size_eval"])*25, quiet=True, start_method="fork" if "fork" in mp.get_all_start_methods() else "spawn",
                                    batch_sampler=MaxTokensBatchSampler(max_tokens=int(run_config["batch_size_eval"])*run_config["max_doc_length"], sorting_keys=["doc_tokens"], padding_noise=0))
    loader.index_with(_vocab)
    return loader


def allennlp_query_gen_train_loader(model_config, run_config, _input_file):
    '''
    Load examples from a .tsv file in the reranking candidate file format: q_id<tab>d_id<tab>q_text<tab>d_text

    (Using allennlp's v2 multiprocess loader)
    '''

    _tokenizer, _token_indexers, _vocab = _get_indexer(model_config, max(run_config["max_doc_length"], run_config["max_query_length"]))


    reader = IndependentReRankingDatasetReader(tokenizer=_tokenizer, token_indexers=_token_indexers,
                                                   max_doc_length=run_config["max_doc_length"], max_query_length=run_config["max_query_length"],
                                                   min_doc_length=run_config.get("min_doc_length",-1), min_query_length=run_config.get("min_query_length",-1),
                                                   query_augment_mask_number=run_config.get("query_augment_mask_number",-1), train_qa_spans=run_config.get("train_qa_spans",False))

    loader = MultiProcessDataLoader(reader, data_path=_input_file, num_workers=run_config["dataloader_num_workers"],
                                    max_instances_in_memory=int(run_config["batch_size_train"])*25, quiet=True, start_method="fork" if "fork" in mp.get_all_start_methods() else "spawn",
                                    batch_size=run_config["batch_size_train"])
    loader.index_with(_vocab)
    return loader


def allennlp_query_gen_inference_loader(model_config, run_config, _input_file,):
    '''
    Load examples from a .tsv file in the single sequence format: id<tab>text and augment it with conditional query codes

    (Using allennlp's v2 multiprocess loader)
    '''

    _tokenizer, _token_indexers, _vocab = _get_indexer(model_config, run_config["max_doc_length"])
    max_length = model_config["max_doc_length"]
    batch_size = run_config["collection_batch_size"]


    reader = ConditionalQueryGenerationInferenceReader(tokenizer=_tokenizer, token_indexers=_token_indexers,
                                                       max_doc_length=max_length,
                                                       target_distribution_file=run_config["target_distribution_file"],
                                                       target_number_of_queries_total=run_config["target_number_of_queries_total"])

    loader = MultiProcessDataLoader(reader, data_path=_input_file, num_workers=run_config["dataloader_num_workers"],
                                        max_instances_in_memory=int(batch_size)*25, quiet=True, start_method="fork" if "fork" in mp.get_all_start_methods() else "spawn",
                                        batch_sampler=MaxTokensBatchSampler(max_tokens=int(batch_size)*max_length, sorting_keys=["doc_tokens"], padding_noise=0))
    loader.index_with(_vocab)
    return loader


def _get_indexer(model_config, max_length):
    # default values
    _tokenizer = BlingFireTokenizer()
    _vocab = Vocabulary()

    if model_config["token_embedder_type"] == "embedding":
        _token_indexers = {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
        _vocab = Vocabulary.from_files(model_config["vocab_directory"])

    elif model_config["token_embedder_type"] == "bert_embedding" or model_config["token_embedder_type"] == "bert_vectors":
        _tokenizer = PretrainedTransformerTokenizer(model_config["bert_pretrained_model"], do_lowercase=True, start_tokens=[], end_tokens=[])
        _ind = PretrainedBertIndexerNoSpecialTokens(pretrained_model=model_config["bert_pretrained_model"], do_lowercase=True, max_pieces=max_length)
        _token_indexers = {"tokens": _ind}

    elif model_config["token_embedder_type"].startswith("bert"):
        model = model_config["bert_pretrained_model"]
        if "facebook/dpr" in model:
            model = "bert-base-uncased"     # should be the right one (judging from paper + huggingface doc)
        _tokenizer = FastTransformerTokenizer(model,
                                              add_unique_ids=model_config.get("colberter_aggregate_unique_ids",False),
                                              uniqueness_type=model_config.get("colberter_aggregate_unique_ids_type",None),
                                              create_global_id=model_config.get("colberter_compress_to_exact_mini_mode", False))
        _token_indexers = None

    if model_config.get("cqg_condition_groups",False):
        _tokenizer._tokenizer.add_tokens([':query_group0',':query_group1',':query_group2',':query_group3',':query_group4',':query_group5'])

    return _tokenizer, _token_indexers, _vocab
