from rich.console import Console
import random
from matchmaker.dataloaders.transformer_tokenizer import *
from allennlp.data.batch import Batch
from matchmaker.utils.core_metrics import *
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.instance import Instance
from allennlp.data.fields import ArrayField
import numpy as np
from allennlp.data.data_loaders.data_loader import TensorDict
from allennlp.data.data_loaders.multiprocess_data_loader import WorkerError
import traceback
from collections import defaultdict
from typing import Any, Dict, Iterator, List
import logging
import torch
import torch.multiprocessing as mp

class PseudoLabelDatasetLoader():
    """
    
    """

    def __init__(
        self,

        query_file: str,
        collection_file: str,
        rankings_with_teacher_scores: str,

        selection_type: str, # values: "scores", "scores-non-fixed", "top-rank"
        min_pos_score: float,
        max_diff_to_be_pos: float,
        min_diff_to_neg: float,

        batch_size: int,

        tokenizer: Tokenizer = None,

        max_doc_length: int = -1,
        max_query_length: int = -1,

        concatenate_sequences = False,
        random_seed=42,
    ):

        self.query_file = query_file
        self.collection_file = collection_file
        self.rankings_with_teacher_scores = rankings_with_teacher_scores
        self.batch_size = batch_size

        self._tokenizer = tokenizer

        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length

        if type(tokenizer) != FastTransformerTokenizer:
            raise Exception("only huggingface tokenizer supported")

        self.selection_type = selection_type
        self.min_pos_score = min_pos_score
        self.max_diff_to_be_pos = max_diff_to_be_pos
        self.min_diff_to_neg = min_diff_to_neg


        self.read_with_scores = True
        self.concatenate_sequences = concatenate_sequences
        self.seed = random_seed

        self.uniqe_pos_only = False

    def __iter__(self) -> Iterator[TensorDict]:
        
        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")

        queue: mp.JoinableQueue = ctx.JoinableQueue(1000)
        worker = ctx.Process(
            target=self.data_loader_subprocess, args=(queue,), daemon=True
        )
        worker.start()

        try:
            for batch, worker_error in iter(queue.get, (None, None)):
                if worker_error is not None:
                    e, tb = worker_error
                    raise WorkerError(e, tb)

                yield batch
                queue.task_done()
        finally:
            if hasattr(queue, "close"):  # for compat with different Python versions.
                queue.close()  # type: ignore[attr-defined]
            if worker.is_alive():
                worker.terminate()

    def load_data(self):

        console = Console()

        console.log("[PseudoLabel] Loading rankings from:",self.rankings_with_teacher_scores)
        self.pos_by_qid = defaultdict(list)
        self.neg_by_qid = defaultdict(list)

        stat_total_pos = 0
        stat_total_neg = 0
        with open(self.rankings_with_teacher_scores, "r", encoding="utf8") as qf:
            current_q_id = ""
            current_top_score = 0
            for line in qf:
                ls = line.split()  # pos_score<t>neg_score<t>pos_id<t>neg_id
                if current_q_id != ls[0]:
                    current_q_id = ls[0]
                    current_top_score = float(ls[3])
                    if self.selection_type == "scores" or self.selection_type == "scores-non-fixed":
                        if current_top_score >= self.min_pos_score:
                            self.pos_by_qid[ls[0]].append((ls[1],float(ls[3])))
                            stat_total_pos+=1

                    elif self.selection_type == "top-rank": 
                        self.pos_by_qid[ls[0]].append((ls[1],float(ls[3])))
                        stat_total_pos+=1
                else:
                    score = float(ls[3])
                    if self.selection_type == "scores":
                        if score >= current_top_score - self.max_diff_to_be_pos and score >= self.min_pos_score:
                            self.pos_by_qid[ls[0]].append((ls[1],score))
                            stat_total_pos+=1

                        elif score < current_top_score - self.min_diff_to_neg:
                            if ls[0] in self.pos_by_qid:
                                self.neg_by_qid[ls[0]].append((ls[1],score))
                                stat_total_neg+=1

                    elif self.selection_type == "scores-non-fixed":
                        if score >= current_top_score - self.max_diff_to_be_pos: # TODO apply this fix and score >= min_pos_score:
                            self.pos_by_qid[ls[0]].append((ls[1],score))
                            stat_total_pos+=1

                        elif score < current_top_score - self.min_diff_to_neg:
                            if ls[0] in self.pos_by_qid:
                                self.neg_by_qid[ls[0]].append((ls[1],score))
                                stat_total_neg+=1

                    elif self.selection_type == "top-rank": 
                        if score >= current_top_score - self.max_diff_to_be_pos:
                            self.pos_by_qid[ls[0]].append((ls[1],score))
                            stat_total_pos+=1

                        elif score < current_top_score - self.min_diff_to_neg:
                            if ls[0] in self.pos_by_qid:
                                self.neg_by_qid[ls[0]].append((ls[1],score))
                                stat_total_neg+=1


        console.log("[PseudoLabel] Loading collection from:",self.collection_file)
        self.collection = {}
        self.collection_ids = []
        with open(self.collection_file, "r", encoding="utf8") as cf:
            for line in cf:
                ls = line.split("\t")  # id<\t>text ....
                self.collection[ls[0]] = ls[1].rstrip()[:100_000]
                self.collection_ids.append(ls[0])

        console.log("[PseudoLabel] Loading queries from:",self.query_file)
        self.queries = {}
        with open(self.query_file, "r", encoding="utf8") as qf:
            for line in qf:
                ls = line.split("\t")  # id<\t>text ....
                self.queries[ls[0]] = ls[1].rstrip()

        self.query_ids = np.array(sorted(list(set(self.pos_by_qid.keys()).intersection(set(self.neg_by_qid.keys())))))

        console.log(f"[PseudoLabel] Done loading! Using {stat_total_pos} positives and {stat_total_neg} negatives for {len(self.query_ids)} queries")

    def data_loader_subprocess(self, queue):

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        try:
            self.load_data()

            query_target_count = self.batch_size # int((self.batch_size / self.clusters_per_batch))

            while len(self.query_ids) > query_target_count:

                main_instances = []

                #while len(main_instances) < self.batch_size:

                #q_ids = random.sample(self.query_ids, query_target_count)
                q_id_idxs = random.sample(range(len(self.query_ids)), query_target_count)
                
                query_idx_remove_buffer = [] # only used for self.uniqe_pos_only==True, we need to buffer the removals, 
                                             # otherwise we break the for loop access of already drawn q_ids

                for q_idx in q_id_idxs:
                    q_id = self.query_ids[q_idx]

                    #if q_id not in self.pos_by_qid or q_id not in self.neg_by_qid: # need to make sure that we did not just remove the query from the dataset (only for self.uniqe_pos_only==True)
                    #    continue

                    pos = random.choice(self.pos_by_qid[q_id])
                    neg = random.choice(self.neg_by_qid[q_id])

                    if self.uniqe_pos_only:
                        self.pos_by_qid[q_id].remove(pos) # ok to remove here, because q_id is unique in this for loop
                        if len(self.pos_by_qid[q_id]) == 0:
                            #del self.pos_by_qid[q_id]
                            query_idx_remove_buffer.append(q_idx)
                            #self.query_ids.pop(q_idx)

                    if self.concatenate_sequences:
                        ret_instance = {
                            "doc_pos_tokens": CustomTransformerTextField(**self._tokenizer.tokenize(self.queries[q_id],self.collection[pos[0]],self.max_query_length + self.max_doc_length)),
                            "doc_neg_tokens": CustomTransformerTextField(**self._tokenizer.tokenize(self.queries[q_id],self.collection[neg[0]],self.max_query_length + self.max_doc_length))}
                    else:
                        ret_instance = {
                            "query_tokens":     self.get_tokenized_query(self.queries[q_id]),
                            "doc_pos_tokens":   self.get_tokenized_document(self.collection[pos[0]]),
                            "doc_neg_tokens":   self.get_tokenized_document(self.collection[neg[0]]),
                        }

                    if self.read_with_scores:
                        ret_instance["pos_score"] = ArrayField(np.array(pos[1]))
                        ret_instance["neg_score"] = ArrayField(np.array(neg[1]))

                    main_instances.append(Instance(ret_instance))

                    #if len(main_instances) == self.batch_size:
                    #    break
                if self.uniqe_pos_only:
                    if len(query_idx_remove_buffer) > 0:
                        self.query_ids = np.delete(self.query_ids,query_idx_remove_buffer)

                main_batch = Batch(main_instances)
                main_batch = main_batch.as_tensor_dict(main_batch.get_padding_lengths())

                queue.put((main_batch,None))

        except Exception as e:
            queue.put((None, (repr(e), traceback.format_exc())))
        
        queue.put((None, None))
        # Wait until this process can safely exit.
        queue.join()

    def get_tokenized_query(self, text):
        query_tokenized = self._tokenizer.tokenize(text, max_length=self.max_query_length)
        if query_tokenized.get('token_type_ids') is not None:
            query_tokenized.pop('token_type_ids')
        return CustomTransformerTextField(**query_tokenized)

    def get_tokenized_document(self, text):
        doc_tokenized = self._tokenizer.tokenize(text, max_length=self.max_doc_length)
        if doc_tokenized.get('token_type_ids') is not None:
            doc_tokenized.pop('token_type_ids')
        return CustomTransformerTextField(**doc_tokenized)


class PseudoLabelTextDatasetLoader():
    """

    """

    def __init__(
            self,

            rankings_with_teacher_scores: str,

            batch_size: int,

            tokenizer: Tokenizer = None,

            max_doc_length: int = -1,
            max_query_length: int = -1,

            concatenate_sequences=False,
            random_seed=42,
    ):

        self.rankings_with_teacher_scores = rankings_with_teacher_scores
        self.batch_size = batch_size

        self._tokenizer = tokenizer

        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length

        if type(tokenizer) != FastTransformerTokenizer:
            raise Exception("only huggingface tokenizer supported")

        self.read_with_scores = True
        self.concatenate_sequences = concatenate_sequences
        self.seed = random_seed

        self.uniqe_pos_only = False

    def __iter__(self) -> Iterator[TensorDict]:

        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")

        queue: mp.JoinableQueue = ctx.JoinableQueue(1000)
        worker = ctx.Process(
            target=self.data_loader_subprocess, args=(queue,), daemon=True
        )
        worker.start()

        try:
            for batch, worker_error in iter(queue.get, (None, None)):
                if worker_error is not None:
                    e, tb = worker_error
                    raise WorkerError(e, tb)

                yield batch
                queue.task_done()
        finally:
            if hasattr(queue, "close"):  # for compat with different Python versions.
                queue.close()  # type: ignore[attr-defined]
            if worker.is_alive():
                worker.terminate()

    def load_data(self):

        console = Console()

        console.log("[PseudoLabel] Loading rankings from:", self.rankings_with_teacher_scores)

        self.triples = []  # query_id pos_id neg_id pos_score neg_score

        with open(self.rankings_with_teacher_scores, "r", encoding="utf8") as qf:
            for line in qf:
                ls = line.split('\t')  # pos_score neg_score query_text pos_text neg_text
                self.triples.append((float(ls[0]), float(ls[1]), ls[2], ls[3], ls[4]))

        console.log(f"[TripleId] Done loading! Using {len(self.triples)} triples")


    def data_loader_subprocess(self, queue):

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        try:
            self.load_data()

            query_target_count = self.batch_size  # int((self.batch_size / self.clusters_per_batch))

            while True:

                main_instances = []

                while len(main_instances) < self.batch_size:

                    pos_score, neg_score, q_text, pos_text, neg_text  = random.choice(self.triples)

                    if self.concatenate_sequences:
                        ret_instance = {
                            "doc_pos_tokens": CustomTransformerTextField(**self._tokenizer.tokenize(q_text, pos_text,
                                                                                                    self.max_query_length + self.max_doc_length)),
                            "doc_neg_tokens": CustomTransformerTextField(**self._tokenizer.tokenize(q_text, neg_text,
                                                                                                    self.max_query_length + self.max_doc_length))}
                    else:
                        ret_instance = {
                            "query_tokens": self.get_tokenized_query(q_text),
                            "doc_pos_tokens": self.get_tokenized_document(pos_text),
                            "doc_neg_tokens": self.get_tokenized_document(neg_text),
                        }

                    if self.read_with_scores:
                        ret_instance["pos_score"] = ArrayField(np.array(pos_score))
                        ret_instance["neg_score"] = ArrayField(np.array(neg_score))

                    main_instances.append(Instance(ret_instance))

                    if len(main_instances) == self.batch_size:
                        break

                main_batch = Batch(main_instances)
                main_batch = main_batch.as_tensor_dict(main_batch.get_padding_lengths())

                queue.put((main_batch, None))

        except Exception as e:
            queue.put((None, (repr(e), traceback.format_exc())))

        queue.put((None, None))
        # Wait until this process can safely exit.
        queue.join()

    def get_tokenized_query(self, text):
        query_tokenized = self._tokenizer.tokenize(text, max_length=self.max_query_length)
        if query_tokenized.get('token_type_ids') is not None:
            query_tokenized.pop('token_type_ids')
        return CustomTransformerTextField(**query_tokenized)

    def get_tokenized_document(self, text):
        doc_tokenized = self._tokenizer.tokenize(text, max_length=self.max_doc_length)
        if doc_tokenized.get('token_type_ids') is not None:
            doc_tokenized.pop('token_type_ids')
        return CustomTransformerTextField(**doc_tokenized)