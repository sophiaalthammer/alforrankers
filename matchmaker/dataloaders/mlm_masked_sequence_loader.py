# based on: https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/seq2seq.py

from typing import Dict
import logging

from overrides import overrides
import numpy as np
import warnings

from rich.console import Console
from matchmaker.dataloaders.transformer_tokenizer import *
from matchmaker.utils.core_metrics import *
import traceback
from typing import Dict, Iterator
import logging

import torch.nn.functional as F

from allennlp.data.data_loaders.data_loader import TensorDict
from allennlp.data.data_loaders.multiprocess_data_loader import WorkerError
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField,MetadataField,ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.batch import Batch
from matchmaker.dataloaders.transformer_tokenizer import FastTransformerTokenizer
from transformers import BertTokenizer, BertTokenizerFast

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
import random
import copy
from collections import defaultdict
import torch
import torch.multiprocessing as mp

@DatasetReader.register("mlm_seq_loader")
class MLMMaskedSequenceDatasetReader(DatasetReader):
    """
    Read a tsv file containing a single sequence <sequence_id>\t<sequence_string>
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_doc_length:int = -1,
                 min_doc_length:int = -1,
                 mlm_mask_whole_words:bool = True,
                 mask_probability:float = 0.1,
                 mlm_mask_replace_probability:float=0.5,
                 mlm_mask_random_probability:float=0.5,
                 bias_sampling_method="None",
                 bias_merge_alpha=0.5,
                 make_multiple_of=8,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
        self.max_seq_length = max_doc_length
        self.min_seq_length = min_doc_length

        self.max_title_length = 30
        self.min_title_length = -1
        self.mask_title = False

        self.token_type = "full"
        if type(tokenizer) == PretrainedTransformerTokenizer or type(tokenizer) == FastTransformerTokenizer:
            self.token_type = "hf"
            self.padding_value = Token(text = "[PAD]", text_id=tokenizer._tokenizer.pad_token_id)
            self.mask_value = Token(text = "[MASK]", text_id=tokenizer._tokenizer.mask_token_id)
            self.cls_value = Token(text = "[CLS]", text_id=tokenizer._tokenizer.cls_token_id) # forgot to add <cls> to the vocabs
        else:
            self.padding_value = Token(text = "@@PADDING@@",text_id=0)
            self.mask_value = Token(text = "[MASK]",text_id=2)
        self.mask_probability = mask_probability
        self.mlm_mask_replace_probability = mlm_mask_replace_probability
        self.mlm_mask_random_probability = mlm_mask_random_probability
        self.mlm_mask_whole_words = mlm_mask_whole_words

        self.bias_sampling_method = bias_sampling_method
        self.bias_merge_alpha = bias_merge_alpha
        self.token_counter = np.ones(tokenizer._tokenizer.vocab_size,dtype=int)
        self.make_multiple_of = make_multiple_of

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r", encoding="utf8") as data_file:
            #logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                if len(line_parts) == 2:
                    seq_id, seq_text = line_parts
                    seq_title = None
                elif len(line_parts) == 3:
                    seq_id, seq_title, seq_text = line_parts
                    if seq_title == "" or seq_text=="":
                        continue
                else:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))

                yield self.text_to_instance(seq_id, seq_text, seq_title)

    @overrides
    def text_to_instance(self, seq_id:str, seq_text:str, seq_title:str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ

        seq_id_field = MetadataField(seq_id)

        seq_tokenized = self._tokenizer.tokenize(seq_text[:10_000])

        if seq_tokenized.get('token_type_ids') is not None:
            seq_tokenized.pop('token_type_ids')

        if self.max_seq_length > -1:
            seq_tokenized = seq_tokenized[:self.max_seq_length]
            #seq_tokenized_orig = seq_tokenized_orig[:self.max_seq_length]
            #mask_binary = mask_binary[:self.max_seq_length]
        if self.min_seq_length > -1 and len(seq_tokenized) < self.min_seq_length:
            seq_tokenized = seq_tokenized + [self.padding_value] * (self.min_seq_length - len(seq_tokenized))
            #seq_tokenized_orig = seq_tokenized_orig + [self.padding_value] * (self.min_seq_length - len(seq_tokenized_orig))
            #mask_binary = mask_binary + [0] * (self.min_seq_length - len(seq_tokenized))
        
        if self.make_multiple_of > -1 and len(seq_tokenized) % self.make_multiple_of != 0:
            seq_tokenized = seq_tokenized + [self.padding_value] * (self.make_multiple_of - len(seq_tokenized) % self.make_multiple_of)


        seq_tokenized_orig = copy.deepcopy(seq_tokenized)
        mask_binary=[0] * len(seq_tokenized)

        suffix = "##" # self._tokenizer.tokenizer._parameters["suffix"]

        if self.token_type == "full":
            for i in range(len(seq_tokenized)):
                if random.uniform(0,1) < self.mask_probability:
                    if random.uniform(0,1) < self.mlm_mask_replace_probability:
                        seq_tokenized[i] = self.mask_value
                    mask_binary[i]= 1
                #else:
                #    mask_binary.append(0)
        else:

            tfs = np.ndarray(len(seq_tokenized_orig))
            for i,t in enumerate(seq_tokenized_orig):
                self.token_counter[t.text_id] += 1
                tfs[i] = self.token_counter[t.text_id]

            tf_class = tfs < np.median(tfs)

            if self.bias_sampling_method == "None":

                for i in range(len(seq_tokenized)):
                    replace_with_mask = False
                    replace_with_random = False
                    if i == 0 or (not self.mlm_mask_whole_words or seq_tokenized_orig[i-1].text.startswith(suffix)): # make sure to start masking at a word start
                        if random.uniform(0,1) < self.mask_probability:
                            if random.uniform(0,1) < self.mlm_mask_replace_probability:
                                replace_with_mask = True
                                seq_tokenized[i] = self.mask_value
                            elif random.uniform(0,1) < self.mlm_mask_random_probability:
                                replace_with_random = True
                                id_ = random.randint(0,self._tokenizer._tokenizer.vocab_size)
                                tok = self._tokenizer._tokenizer.convert_ids_to_tokens(id_)
                                seq_tokenized[i] = Token(text = tok, text_id=id_)

                            mask_binary[i] = 1
                            if self.mlm_mask_whole_words and not seq_tokenized_orig[i].text.startswith(suffix): # mask until end of full word 
                                for t in range(i+1,len(seq_tokenized)):
                                    if replace_with_mask == True:
                                        seq_tokenized[t] = self.mask_value
                                    elif replace_with_random == True:
                                        id_ = random.randint(0,self._tokenizer._tokenizer.vocab_size)
                                        tok = self._tokenizer._tokenizer.convert_ids_to_tokens(id_)
                                        seq_tokenized[t] = Token(text = tok, text_id=id_)                                    
                                    
                                    mask_binary[t] = 1
                                    if seq_tokenized_orig[t].text.startswith(suffix):
                                        break

            elif self.bias_sampling_method == "tf" or self.bias_sampling_method == "log-tf":

                if self.bias_sampling_method == "log-tf":
                    tfs = np.log2(tfs)

                probability = tfs.sum()/tfs
                probability /= probability.max()
                probability *= self.mask_probability
                #probability[probability < 0.0001] = 0.0001

                probability = probability * (self.mask_probability/(probability.mean()))
                probability[probability > 0.9] = 0.9

                #probability = probability * (self.mask_probability/(probability.mean()))
                #probability[probability > 0.9] = 0.9

                masks = torch.bernoulli(torch.from_numpy(probability))
                for i in range(len(seq_tokenized)):
                    if masks[i] == 1:
                        replace_with_mask = False
                        if random.uniform(0,1) < self.mlm_mask_replace_probability:
                            replace_with_mask = True
                            seq_tokenized[i] = self.mask_value
                        mask_binary[i] = 1

                        # opt 1 - previous tokens are part of the word -> mask them also
                        if i > 0 and not seq_tokenized_orig[i-1].text.endswith(suffix):
                            for t in list(range(0,i-1))[::-1]:
                                if replace_with_mask == True:
                                    seq_tokenized[t] = self.mask_value
                                mask_binary[t] = 1
                                if seq_tokenized_orig[t].text.endswith(suffix):
                                    break

                        # opt 2 - next tokens are part of the word -> mask them also
                        if not seq_tokenized_orig[i].text.endswith(suffix): # mask until end of full word 
                            for t in range(i+1,len(seq_tokenized)):
                                if replace_with_mask == True:
                                    seq_tokenized[t] = self.mask_value
                                mask_binary[t] = 1
                                if seq_tokenized_orig[t].text.endswith(suffix):
                                    break
        
        seq_field = TextField(seq_tokenized, self._token_indexers)
        seq_field_orig = TextField(seq_tokenized_orig, self._token_indexers)

        if seq_title != None:

            title_tokenized = self._tokenizer.tokenize(seq_title)

            if self.max_title_length > -1:
                title_tokenized = title_tokenized[:self.max_title_length]
            if self.min_title_length > -1 and len(title_tokenized) < self.min_title_length:
                title_tokenized = title_tokenized + [self.padding_value] * (self.min_title_length - len(title_tokenized))

            #if self.make_multiple_of > -1 and len(seq_tokenized) % self.make_multiple_of != 0:
            #    seq_tokenized = seq_tokenized + [self.padding_value] * (self.make_multiple_of - len(seq_tokenized) % self.make_multiple_of)

            title_tokenized.insert(0,self.cls_value)

            title_tokenized_masked = copy.deepcopy(title_tokenized)
            title_mask_binary=[0] * len(title_tokenized_masked)

            for i in range(len(title_tokenized_masked)):
                if random.uniform(0,1) < self.mask_probability:
                    if random.uniform(0,1) < self.mlm_mask_replace_probability:
                        replace_with_mask = True
                        title_tokenized_masked[i] = self.mask_value
                    title_mask_binary[i] = 1

            title_field = TextField(title_tokenized, self._token_indexers)
            title_field_masked = TextField(title_tokenized_masked, self._token_indexers)

            return Instance({
                "seq_id":seq_id_field,
                "title_tokens":title_field_masked,
                "title_tokens_original":title_field,
                "title_tokens_mask":ArrayField(np.array(title_mask_binary)),
                "seq_masked":ArrayField(np.array(mask_binary)),
                "seq_tf_info":ArrayField(np.array(tf_class)),
                "seq_tokens":seq_field,
                "seq_tokens_original":seq_field_orig})
        else:
            return Instance({
                "seq_id":seq_id_field,
                "seq_masked":ArrayField(np.array(mask_binary)),
                "seq_tf_info":ArrayField(np.array(tf_class)),
                "seq_tokens":seq_field,
                "seq_tokens_original":seq_field_orig})


class MLMDatasetLoader():
    """

    """

    def __init__(
            self,
            collection_file: str,
            batch_size: int,
            whole_word_masking: True,
            tokenizer: Tokenizer = None,
            max_doc_length: int = -1,
            random_seed=42,
            min_doc_length: int = -1,
            mlm_mask_whole_words: bool = True,
            mask_probability: float = 0.15,
            mlm_mask_replace_probability: float = 0.8,
            mlm_mask_random_probability: float = 0.1,
            bias_sampling_method="None",
            bias_merge_alpha=0.5,
            make_multiple_of=-1,
            random_spans = False,
            tasb = False,
            tasb_cluster_file = None,
            tasb_weight = 1,
            grad_acc = -1,
            cached_chunk_size = -1
    ):

        self.collection_file = collection_file
        self.batch_size = batch_size

        self._tokenizer = tokenizer

        self.max_doc_length = max_doc_length

        if type(tokenizer) != FastTransformerTokenizer:
            raise Exception("only huggingface tokenizer supported")

        self.token_counter = np.ones(tokenizer._tokenizer.vocab_size, dtype=int)
        self.bias_sampling_method = bias_sampling_method
        self.bias_merge_alpha = bias_merge_alpha

        self.seed = random_seed

        # mlm masking
        self.min_doc_length = min_doc_length
        self.mlm_mask_whole_words = mlm_mask_whole_words
        self.mask_probability = mask_probability
        self.mlm_mask_replace_probability = mlm_mask_replace_probability
        self.mlm_mask_random_probability = mlm_mask_random_probability
        self.make_multiple_of = make_multiple_of

        self.whole_word_masking = whole_word_masking

        # cocondenser random spans?
        self.random_spans = random_spans
        self.tasb = tasb
        if self.tasb:
            self.tasb_cluster_file = tasb_cluster_file
            self.tasb_weight = tasb_weight
            self.grad_acc = grad_acc

        self.cached_chunk_size = cached_chunk_size

        self.token_type = "full"
        if type(tokenizer) == PretrainedTransformerTokenizer or type(tokenizer) == FastTransformerTokenizer:
            self.token_type = "hf"
            self.padding_value = Token(text="[PAD]", text_id=tokenizer._tokenizer.pad_token_id)
            self.mask_value = Token(text="[MASK]", text_id=tokenizer._tokenizer.mask_token_id)
            self.cls_value = Token(text="[CLS]",
                                   text_id=tokenizer._tokenizer.cls_token_id)  # forgot to add <cls> to the vocabs
        else:
            self.padding_value = Token(text="@@PADDING@@", text_id=0)
            self.mask_value = Token(text="[MASK]", text_id=2)


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

        console.log("[MLMPseudoLabel] Loading rankings from:", self.collection_file)

        self.collection = {}
        self.collection_ids = []
        with open(self.collection_file, "r", encoding="utf8") as cf:
            for line in cf:
                ls = line.split("\t")  # id<\t>text ....
                self.collection[ls[0]] = ls[1].rstrip()
                self.collection_ids.append(ls[0])

        console.log(f"[MLMLabel] Done loading of the collection!")

        if self.tasb:
            console.log("[TASBalanced] Loading cluster assignments from:", self.tasb_cluster_file)
            self.query_clusters = []
            all_cluster_ids = []
            with open(self.tasb_cluster_file, "r", encoding="utf8") as qf:
                for line in qf:
                    ls = line.split()  # id<\t>id ....
                    self.query_clusters.append(ls)
                    all_cluster_ids.extend(ls)

            self.query_ids = set(self.collection_ids).intersection(set(all_cluster_ids))

            # clean clusters, to only have matching ids with pair file
            for i, c in enumerate(self.query_clusters):
                self.query_clusters[i] = list(set(c).intersection(self.query_ids))
            self.query_clusters = [c for c in self.query_clusters if len(c) > 0]

            console.log("[TASBalanced] Done loading! Using ", len(self.query_ids), " queries from ",
                        len(self.query_clusters), "clusters for seed:", self.seed)

    def data_loader_subprocess(self, queue):

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        try:
            self.load_data()

            doc_target_count = self.batch_size  # int((self.batch_size / self.clusters_per_batch))
            grad_acc_step = 0

            while len(self.collection_ids) > doc_target_count:

                main_instances = []

                if self.tasb:
                    if self.grad_acc > -1:
                        if grad_acc_step == 0:
                            c_idx = random.randint(0, len(self.query_clusters) - 1)
                            current_cluster_id = c_idx
                            grad_acc_step = 1
                        elif 0 < grad_acc_step < (self.grad_acc-1):
                            c_idx = current_cluster_id
                            grad_acc_step += 1
                        else:
                            c_idx = current_cluster_id
                            grad_acc_step = 0
                    elif self.cached_chunk_size > -1:
                        if grad_acc_step == 0:
                            c_idx = random.randint(0, len(self.query_clusters) - 1)
                            current_cluster_id = c_idx
                            grad_acc_step = 1
                        elif 0 < grad_acc_step < (self.cached_chunk_size-1):
                            c_idx = current_cluster_id
                            grad_acc_step += 1
                        else:
                            c_idx = current_cluster_id
                            grad_acc_step = 0
                    else:
                        c_idx = random.randint(0, len(self.query_clusters) - 1)
                    print('current cluster id: {}'.format(c_idx))
                    # take a query sample out of that cluster
                    if self.tasb_weight > 0:
                        if int(np.floor(doc_target_count*self.tasb_weight)) < len(self.query_clusters[c_idx]):
                            c_id_idxs_tasb = random.sample(self.query_clusters[c_idx],
                                                           int(np.floor(doc_target_count*self.tasb_weight)))
                        else:
                            c_id_idxs_tasb = self.query_clusters[c_idx]

                        c_id_idxs_random = random.sample(range(len(self.collection_ids)), doc_target_count-len(c_id_idxs_tasb))

                        c_id_idxs_tasb.extend(c_id_idxs_random)
                        c_id_idxs = c_id_idxs_tasb
                    else:
                        c_id_idxs = random.sample(range(len(self.collection_ids)), doc_target_count)
                else:
                    c_id_idxs = random.sample(range(len(self.collection_ids)), doc_target_count)

                query_idx_remove_buffer = []  # only used for self.uniqe_pos_only==True, we need to buffer the removals,
                # otherwise we break the for loop access of already drawn q_ids

                # include padding!!
                for c_idx in c_id_idxs:
                    c_id = self.collection_ids[int(c_idx)]

                    seq_id_field = MetadataField(c_id)

                    seq_tokenized_field = self.get_tokenized_document(self.collection[c_id], random_spans=self.random_spans)
                    seq_tokenized_orig_field = copy.deepcopy(seq_tokenized_field)
                    seq_labels_field = copy.deepcopy(seq_tokenized_field)

                    seq_tokenized = seq_tokenized_field.input_ids
                    seq_tokenized_labels = seq_tokenized_orig_field.input_ids

                    mask_binary = torch.zeros(len(seq_tokenized))

                    if self.whole_word_masking:
                        ref_tokens = []
                        for i in range(len((seq_tokenized))):
                            token = self._tokenizer._tokenizer._convert_id_to_token(seq_tokenized[i])
                            ref_tokens.append(token)
                        mask_labels = self._whole_word_mask(ref_tokens)
                        for i in range(len(seq_tokenized)):
                            if mask_labels[i] == 1:
                                replace_probability = random.uniform(0, 1)
                                # 80% are replaces with mask
                                if replace_probability < self.mlm_mask_replace_probability:
                                    seq_tokenized[i] = 103  # thats the id for mask #self.mask_value
                                # 10% a random token
                                elif self.mlm_mask_replace_probability < replace_probability < \
                                        (self.mlm_mask_replace_probability + self.mlm_mask_random_probability):
                                    # 10% are not replaced
                                    seq_tokenized[i] = random.sample(range(self._tokenizer._tokenizer.vocab_size), 1)[0]  #30522 # include random token id, but here dependent on vocab size
                                mask_binary = 1
                            else:
                                # for those the mlm loss is not computed!
                                seq_tokenized_labels[i] = -100

                    else:
                        for i in range(len(seq_tokenized)):
                            if random.uniform(0, 1) < self.mask_probability:
                                replace_probability = random.uniform(0, 1)
                                # 80% are replaces with mask
                                if replace_probability < self.mlm_mask_replace_probability:
                                    seq_tokenized[i] = 103 # thats the id for mask #self.mask_value
                                # 10% a random token
                                elif self.mlm_mask_replace_probability < replace_probability < \
                                        (self.mlm_mask_replace_probability + self.mlm_mask_random_probability):
                                # 10% are not replaced
                                    seq_tokenized[i] = random.sample(range(self._tokenizer._tokenizer.vocab_size), 1)[0] # include random token id, but here dependent on vocab size
                                mask_binary = 1
                            else:
                                # for those the mlm loss is not computed!
                                seq_tokenized_labels[i] = -100

                    seq_tokenized_field.input_ids = seq_tokenized
                    seq_labels_field.input_ids = seq_tokenized_labels
                    # maybe i also need to change the whole ids for the cocolberter pretraining

                    ret_instance = {
                        "seq_id": seq_id_field,
                        "seq_masked": ArrayField(np.array(mask_binary)),
                        "seq_tokens": seq_tokenized_field,
                        "seq_tokens_original": seq_tokenized_orig_field,
                        "seq_labels": seq_labels_field}

                    main_instances.append(Instance(ret_instance))

                main_batch = Batch(main_instances)
                main_batch = main_batch.as_tensor_dict(main_batch.get_padding_lengths())

                queue.put((main_batch, None))

        except Exception as e:
            queue.put((None, (repr(e), traceback.format_exc())))

        queue.put((None, None))
        # Wait until this process can safely exit.
        queue.join()

    def get_tokenized_document(self, text, random_spans=False):
        doc_tokenized = self._tokenizer.tokenize(text, max_length=self.max_doc_length, padding=True, random_spans=random_spans)
        if doc_tokenized.get('token_type_ids') is not None:
            doc_tokenized.pop('token_type_ids')
        return CustomTransformerTextField(**doc_tokenized)

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        if not isinstance(self._tokenizer, (BertTokenizer, BertTokenizerFast)):
            warnings.warn(
                "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
                "Please refer to the documentation for more information."
            )

        cand_indexes = []
        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * self.mask_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels