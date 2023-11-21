from typing import Dict

from overrides import overrides
import numpy as np
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField,MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.token_indexers import TokenIndexer
from matchmaker.dataloaders.transformer_tokenizer import *
from rich.console import Console

from typing import List

token_groups=[
        ["what","how","where","who","when","which","why","was","is","can","does","do","are","define","definition","meaning","mean"],
        ["---other---"]
    ]

def get_type_idx_for_query(query:List[str]):
    found = False
    found_i = len(token_groups) - 1
    for i,g_arr in enumerate(token_groups):
        for token in query:
            if token in g_arr:
                found = True
                found_i = i
                break
        if found: 
            break
    return found_i

def approximate_target_distribution(queries:List[str]):
    """
    Given a list of queries, return a list of approximate target distributions.
    """
    
    max_length = 30


    length_by_token_group = np.zeros((len(token_groups),max_length))
    labels_type = np.array([[0]*max_length, [1]*max_length]).flatten()
    labels_length = np.array([np.arange(1,max_length+1),np.arange(1,max_length+1)]).flatten()

    for q in queries:
        toks = q.split()
        tok_len = min(len(toks),max_length)

        found_i = get_type_idx_for_query(toks)

        length_by_token_group[found_i][tok_len - 1] += 1

    #print(length_by_token_group)
    length_by_token_group = length_by_token_group / length_by_token_group.sum()
    #print(length_by_token_group)
    return length_by_token_group.flatten(), (labels_type, labels_length)

def approximate_target_distribution_from_file(filename:str):
    queries = []
    with open(filename,"r",encoding="utf8") as query_file:
        for line in query_file:
            ls = line.split("\t") # two possible: id\tquery or id\t\id\query\tpassage

            if len(ls) == 2:
                all_chars = ls[1].strip()
            elif len(ls) == 4:
                all_chars = ls[2].strip()
            queries.append(all_chars)

    return approximate_target_distribution(queries)



class ConditionalQueryGenerationInferenceReader(DatasetReader):
    """
    Read a tsv file containing a passage collection.
    
    Expected format for each input line: <doc_id>\t<doc_sequence_string>
    The output of ``read`` is a list of ``Instance`` s with the fields:
        doc_tokens: ``TextField`` 
        target_query_type: ``MetadataField``
        target_query_length: ``MetadataField``


    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. 
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 
                 max_doc_length:int = -1,
                 max_query_length:int = -1,

                 target_distribution_file:str = None,
                 target_number_of_queries_total:int = 1 # ATTENTION, this is per worker!! (divide on your own if using > 1 worker)
                 ):

        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True
        )
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers

        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length

        self.target_number_of_queries_total = target_number_of_queries_total

        target_distribution,(target_label_types,target_label_lengths) = approximate_target_distribution_from_file(target_distribution_file)

        console = Console()

        console.log("[QueryGenLoader] Targeting distribution:",target_distribution*target_number_of_queries_total,", labels",(target_label_types,target_label_lengths))

        self.target_distribution = target_distribution
        self.target_label_types = target_label_types
        self.target_label_lengths = target_label_lengths

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r", encoding="utf8") as data_file:
            #logger.info("Reading instances from lines in file at: %s", file_path)
            for i,line in enumerate(self.shard_iterable(data_file)):
                if i == self.target_number_of_queries_total:
                    break

                line = line.strip()

                if not line:
                    continue

                line_parts = line.split('\t')
                if len(line_parts) == 2:
                    doc_id, doc_sequence = line_parts
                else:
                    raise ConfigurationError("Invalid line format: %s" % (line))

                yield self.text_to_instance(doc_id, doc_sequence)

    @overrides
    def text_to_instance(self,  doc_id:str, doc_sequence: str) -> Instance:

        doc_id_field = MetadataField(doc_id)

        target_idx = np.random.choice(len(self.target_distribution),1,replace=False,p=self.target_distribution)[0]

        concat_sequence = (":query_group"+str(self.target_label_types[target_idx]) + " "+ str(self.target_label_lengths[target_idx]) + " " + doc_sequence)

        doc_tokenized = self._tokenizer.tokenize(concat_sequence, max_length=self.max_doc_length)
        if doc_tokenized.get('token_type_ids') is not None:
            doc_tokenized.pop('token_type_ids')
        doc_field = TransformerTextField(**doc_tokenized,padding_token_id=self._tokenizer._tokenizer.pad_token_id)

        return Instance({
            "doc_id":doc_id_field,
            "doc_tokens":doc_field,
            "target_query_type":MetadataField(self.target_label_types[target_idx]),
            "target_query_length":MetadataField(self.target_label_lengths[target_idx])})


