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
#mp.set_sharing_strategy("file_system")  # VERY MUCH needed for linux !! makes everything MUCH faster


class TripleIdDatasetLoader():
    """
    
    """

    def __init__(
        self,

        query_file: str,
        collection_file: str,
        triples_with_teacher_scores: str,

        batch_size: int,

        tokenizer: Tokenizer = None,

        max_doc_length: int = -1,
        max_query_length: int = -1,

        concatenate_sequences = False,
        random_seed=42,
    ):

        self.query_file = query_file
        self.collection_file = collection_file
        self.triples_with_teacher_scores = triples_with_teacher_scores
        self.batch_size = batch_size

        self._tokenizer = tokenizer

        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length

        if type(tokenizer) != FastTransformerTokenizer:
            raise Exception("only huggingface tokenizer supported")

        self.read_with_scores = True
        self.concatenate_sequences = concatenate_sequences
        self.seed = random_seed

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

        console.log("[TripleId] Loading rankings from:",self.triples_with_teacher_scores)
        self.triples = [] # query_id pos_id neg_id pos_score neg_score

        with open(self.triples_with_teacher_scores, "r", encoding="utf8") as qf:
            for line in qf:
                ls = line.split() # pos_score neg_score query_id pos_id neg_id
                self.triples.append((ls[2],ls[3],ls[4],float(ls[0]),float(ls[1])))

        console.log("[TripleId] Loading collection from:",self.collection_file)
        self.collection = {}
        self.collection_ids = []
        with open(self.collection_file, "r", encoding="utf8") as cf:
            for line in cf:
                ls = line.split("\t")  # id<\t>text ....
                self.collection[ls[0]] = ls[1].rstrip()[:100_000]
                self.collection_ids.append(ls[0])

        console.log("[TripleId] Loading queries from:",self.query_file)
        self.queries = {}
        with open(self.query_file, "r", encoding="utf8") as qf:
            for line in qf:
                ls = line.split("\t")  # id<\t>text ....
                self.queries[ls[0]] = ls[1].rstrip()

        console.log(f"[TripleId] Done loading! Using {len(self.triples)} triples")

    def data_loader_subprocess(self, queue):

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        try:
            self.load_data()

            query_target_count = self.batch_size # int((self.batch_size / self.clusters_per_batch))

            while True:

                main_instances = []

                while len(main_instances) < self.batch_size:

                    q_id,pos_id,neg_id,pos_score,neg_score = random.choice(self.triples)

                    if self.concatenate_sequences:
                        ret_instance = {
                            "doc_pos_tokens": CustomTransformerTextField(**self._tokenizer.tokenize(self.queries[q_id],self.collection[pos_id],self.max_query_length + self.max_doc_length)),
                            "doc_neg_tokens": CustomTransformerTextField(**self._tokenizer.tokenize(self.queries[q_id],self.collection[neg_id],self.max_query_length + self.max_doc_length))}
                    else:
                        ret_instance = {
                            "query_tokens":     self.get_tokenized_query(self.queries[q_id]),
                            "doc_pos_tokens":   self.get_tokenized_document(self.collection[pos_id]),
                            "doc_neg_tokens":   self.get_tokenized_document(self.collection[neg_id]),
                        }

                    if self.read_with_scores:
                        ret_instance["pos_score"] = ArrayField(np.array(pos_score))
                        ret_instance["neg_score"] = ArrayField(np.array(neg_score))

                    main_instances.append(Instance(ret_instance))

                    if len(main_instances) == self.batch_size:
                        break

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