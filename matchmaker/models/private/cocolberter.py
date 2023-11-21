from collections import namedtuple
from typing import Dict, Union

import warnings
import os
import torch
from torch import nn as nn
from torch import Tensor

from transformers import PreTrainedModel
from transformers.modeling_utils import PreTrainedModel as PreTrainedModelUtils
from transformers import AutoModel
from transformers import BertModel, BertConfig, AutoModel, AutoModelForMaskedLM, AutoConfig, PretrainedConfig, \
    RobertaModel
from transformers.models.distilbert.modeling_distilbert import TransformerBlock, DistilBertForMaskedLM, DistilBertModel
from transformers.models.bert.modeling_bert import BertPooler, BertOnlyMLMHead, BertPreTrainingHeads, BertLayer
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPooling, MaskedLMOutput
from transformers.activations import *


class CoColBERTerConfig(PretrainedConfig):
    model_type = "ColBERT"
    bert_model: str
    dropout: float = 0.0
    return_vecs: bool = False
    dual_loss: bool = False
    trainable: bool = True

    compression_dim: int = -1
    use_contextualized_stopwords = False
    aggregate_unique_ids = False
    retrieval_compression_dim = -1

    compress_to_exact_mini_mode = False
    second_compress_dim = -1
    co_n_head_layers = -1
    skip_from = 1


class CoColBERTer(PreTrainedModel):
    """
    CoColBERTer model
    """

    config_class = CoColBERTerConfig
    base_model_prefix = "bert_model"

    # is_teacher_model = False  # gets overwritten by the dynamic teacher runner

    @staticmethod
    def from_config(config):
        cfg = CoColBERTerConfig()
        cfg.bert_model = config["bert_pretrained_model"]
        cfg.return_vecs = config.get("in_batch_negatives", False)
        cfg.dual_loss = config.get("train_dual_loss", False)
        cfg.trainable = config["bert_trainable"]

        cfg.compression_dim = config["colberter_compression_dim"]
        cfg.retrieval_compression_dim = config["colberter_retrieval_compression_dim"]
        cfg.use_contextualized_stopwords = config["colberter_use_contextualized_stopwords"]
        cfg.aggregate_unique_ids = config["colberter_aggregate_unique_ids"]
        cfg.compress_to_exact_mini_mode = config.get("colberter_compress_to_exact_mini_mode", False)
        cfg.second_compress_dim = config.get("colberter_second_compress_dim", -1)

        # condenser parameters
        cfg.co_n_head_layers = config.get("co_n_head_layers", -1)
        cfg.skip_from = config.get("skip_from", 2)
        cfg.token_embedding_weight = config.get("token_embedding_weight", -1)

        return CoColBERTer(cfg)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.bert_model.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.bert_model.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def __init__(self,
                 cfg) -> None:
        super().__init__(cfg)

        self.return_vecs = cfg.return_vecs
        self.dual_loss = cfg.dual_loss

        self.bert_model = AutoModel.from_pretrained(cfg.bert_model)
        # previously this was AutoModelForMaskedMLM.from_pretrained(cfg.bert_model)
        self.mlm_loss_fct = nn.CrossEntropyLoss()

        # only mlm head in forward integrate!
        self.use_compressor = cfg.compression_dim > -1
        if self.use_compressor:
            self.activation_compr = get_activation(self.bert_model.config.activation)
            self.vocab_transform_compr = nn.Linear(cfg.compression_dim, cfg.compression_dim)
            self.vocab_layer_norm_compr = nn.LayerNorm(cfg.compression_dim, eps=1e-12)
            self.vocab_projector_compr = nn.Linear(cfg.compression_dim, self.bert_model.config.vocab_size)

            self.vocab_transform_compr.apply(self.bert_model._init_weights)
            self.vocab_layer_norm_compr.apply(self.bert_model._init_weights)
            self.vocab_projector_compr.apply(self.bert_model._init_weights)

        self.activation = get_activation(self.bert_model.config.activation)
        self.vocab_transform = nn.Linear(self.bert_model.config.dim, self.bert_model.config.dim)
        self.vocab_layer_norm = nn.LayerNorm(self.bert_model.config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(self.bert_model.config.dim, self.bert_model.config.vocab_size)

        self.vocab_transform.apply(self.bert_model._init_weights)
        self.vocab_layer_norm.apply(self.bert_model._init_weights)
        self.vocab_projector.apply(self.bert_model._init_weights)

        self.vocab_size = self.bert_model.config.vocab_size

        for p in self.bert_model.parameters():
            p.requires_grad = cfg.trainable

        if self.use_compressor:
            self.compressor = torch.nn.Linear(self.bert_model.config.hidden_size, cfg.compression_dim)
            # need a decompressor for tokens
            self.decompressor = torch.nn.Linear(cfg.compression_dim, self.bert_model.config.hidden_size)

        self.use_retrieval_compression = cfg.retrieval_compression_dim > -1
        if self.use_retrieval_compression:
            self.compressor_retrieval = torch.nn.Linear(self.bert_model.config.hidden_size,
                                                        cfg.retrieval_compression_dim)
            self.decompressor_retrieval = torch.nn.Linear(cfg.retrieval_compression_dim,
                                                          self.bert_model.config.hidden_size)

        self.aggregate_unique_ids = cfg.aggregate_unique_ids

        # inserted from condenser
        self.use_co_n_head_layers = cfg.co_n_head_layers > -1
        if self.use_co_n_head_layers:
            self.c_head = nn.ModuleList(
                [TransformerBlock(self.bert_model.config) for _ in range(cfg.co_n_head_layers)]) #BertLayer
            self.c_head.apply(self.bert_model._init_weights)   #_init_weights()

        self.skip_from = cfg.skip_from

        self.token_embedding_weight = cfg.token_embedding_weight

        #self.use_contextualized_stopwords = cfg.use_contextualized_stopwords
        #if self.use_contextualized_stopwords:
        #    self.stop_word_reducer = nn.Linear(cfg.compression_dim, 1, bias=True)
        #    torch.nn.init.constant_(self.stop_word_reducer.bias, 1)  # make sure we don't start in a broken state

        #self.compress_to_exact_mini_mode = cfg.compress_to_exact_mini_mode
        #self.use_second_compression = cfg.second_compress_dim > -1
        #if self.use_second_compression:
        #    self.mini_compressor = torch.nn.Linear(cfg.compression_dim, cfg.second_compress_dim)


    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                labels, #: Dict[str, torch.LongTensor]
                use_fp16: bool = True,
                output_secondary_output: bool = False) -> torch.Tensor:

        with torch.cuda.amp.autocast(enabled=use_fp16):
            retrieval_vec, vecs, mask, output, mlm_loss = self.forward_representation(tokens, labels)


            if self.use_retrieval_compression:
                cls_hiddens = self.decompressor_retrieval(retrieval_vec)
            else:
                cls_hiddens = retrieval_vec

            # maybe i need to include that in the forward_shared_encoding where the maskedlmoutput happens!
            if self.use_compressor:
                vecs = self.decompressor(vecs)

            # cls_hiddens2 is the original implementation from cocondenser but without the compression
            #cls_hiddens2 = lm_out.hidden_states[-1][:, :1]
            #cls_hiddens = cls_hiddens2
            skip_hiddens = output.hidden_states[self.skip_from]

            #print(skip_hiddens)
            #print("skip hiddens nan" if torch.any(torch.isnan(skip_hiddens)) == True else "")

            #cls_hiddens = cls_hiddens.unsqueeze(1)

            hiddens = torch.cat([cls_hiddens, skip_hiddens[:, 1:]], dim=1)


            for layer in self.c_head:
                layer_out = layer(
                    hiddens,
                    mask,
                )
                hiddens = layer_out[0]


            # cls does not exist for distilbert
            #scores = self.bert_model.cls(hiddens)
            # is this score only the output from the pretrainig head?
            #https: // github.com / huggingface / transformers / blob / 34097
            #b3304d79ace845316d4929220623279c8bc / src / transformers / models / bert / modeling_bert.py  # L721

            # distilbertformaskedlm https://github.com/huggingface/transformers/blob/main/src/transformers/models/distilbert/modeling_distilbert.py
            prediction_logits = self.vocab_transform(hiddens)  # (bs, seq_length, dim)
            prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
            scores = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)

            return scores, mlm_loss


    def forward_representation(self,
                               tokens: Dict[str, torch.LongTensor],
                               labels: Dict[str, torch.LongTensor]) -> torch.Tensor:

        retrieval_vec, vecs, mask, output, mlm_loss = self.forward_shared_encoding(tokens, labels)

        return retrieval_vec, vecs, mask, output, mlm_loss


    def forward_shared_encoding(self,  # type: ignore
                                tokens: Dict[str, torch.LongTensor],
                                labels: Dict[str, torch.LongTensor]):
        #
        # ColBERT-style encoding
        #
        mask = tokens["attention_mask"].bool()
        output = self.bert_model(
                                # **tokens,
                                 input_ids=tokens["input_ids"],
                                 attention_mask=tokens["attention_mask"],
                                 output_hidden_states=True,
                                 return_dict=True)

        vecs = output[0]

        #
        # ColBERTer: Enhanced Reduction
        # ------------------------------------

        #
        # Part 1: Retrieval, Compress [CLS] vectors only to self.compressor_retrieval dim
        #
        if self.use_retrieval_compression:
            retrieval_vec = self.compressor_retrieval(output.hidden_states[-1][:, :1]) #vecs[:, 0, :]
        else:
            retrieval_vec = output.hidden_states[-1][:, :1]   #vecs[:, 0, :] = output[0][:, 0, :]

        #
        # Part 2, Refinement 2.1: Dimensionality Reduction of Compressed Sub-Word Representations
        # -> seems counter-intuitive to do first, but saves a lot of memory consumption for the aggregation
        #
        if self.use_compressor:
            vecs = self.compressor(vecs)

        #
        # Part 2, Refinement 2.2: Aggregate unique-whole-words
        #
        if self.aggregate_unique_ids:
            #print("use aggregate unique_ids")

            # aggregate whole-words
            aggregation_mask = (tokens["unique_words"].unsqueeze(-1) == tokens["input_ids_to_words_map"].unsqueeze(
                1)).unsqueeze(-1)
            aggregated_vecs = (vecs.unsqueeze(1).expand(-1, aggregation_mask.shape[1], -1, -1) * aggregation_mask).sum(
                2)

            # mean pooling (instead of sum)
            # maybe i need to change the aggregation?
            aggregated_vecs = aggregated_vecs / aggregation_mask.float().sum(-2)

            whole_word_mask = tokens["unique_words"] > 0
            #vecs = aggregated_vecs

            # include deaggregation here!

            # how to deaggregate? with learned weights i would say, and the positions need to be given by the attention mask
            deaggregated_vecs = (aggregated_vecs.unsqueeze(2).expand(-1, -1, aggregation_mask.shape[2], -1) * aggregation_mask).sum(1)

            if self.token_embedding_weight > -1:
                deaggregated_vecs = deaggregated_vecs + self.token_embedding_weight * vecs

            vecs = deaggregated_vecs

            # and then the decompression step as well?
            # and then the masked language modelling loss?

            # here add token embedding additionally? how can i deaggregate it?

        if self.use_compressor:
            prediction_logits = self.vocab_transform_compr(vecs)  # (bs, seq_length, dim)
            prediction_logits = self.activation_compr(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_layer_norm_compr(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_projector_compr(prediction_logits)
        else:
            # here the mlm loss, so that it takes the right vecs, in case I use compression before or aggregation!
            prediction_logits = self.vocab_transform(vecs)  # (bs, seq_length, dim)
            prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_projector(prediction_logits)

        mlm_loss = None
        if labels is not None:
            # -100 index = padding token
            mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)),
                                         labels["input_ids"].view(-1))
        #print('mlm loss is computed!')

        # here return, back to subtokens
        # then back from compression, to original input i think, then mlm loss? then whole word autoencoding?
        # whole word contrastive learning?

        return retrieval_vec, vecs, mask, output, mlm_loss


    def get_param_stats(self):
        return "ColBERTer: "

    def get_param_secondary(self):
        return {}

    def get_output_dim(self):
        dim_info = namedtuple('dim_info', ['cls_vector', "token_vector"])

        if hasattr(self.bert_model.config, "dim"):
            cls_vec_dim = self.bert_model.config.dim
            token_vec_dim = self.bert_model.config.dim
        else:
            cls_vec_dim = self.bert_model.config.hidden_size
            token_vec_dim = self.bert_model.config.hidden_size

        #if self.use_second_compression:
        #    token_vec_dim = self.mini_compressor.out_features
        if self.use_compressor:
            token_vec_dim = self.compressor.out_features

        if self.use_retrieval_compression:
            cls_vec_dim = self.compressor_retrieval.out_features

        return dim_info(cls_vector=cls_vec_dim, token_vector=token_vec_dim)



class CoCoColBERTerConfig(PretrainedConfig):
    model_type = "ColBERT"
    bert_model: str
    dropout: float = 0.0
    return_vecs: bool = False
    dual_loss: bool = False
    trainable: bool = True

    compression_dim: int = -1
    use_contextualized_stopwords = False
    aggregate_unique_ids = False
    retrieval_compression_dim = -1

    compress_to_exact_mini_mode = False
    second_compress_dim = -1
    co_n_head_layers = -1
    skip_from = 1

    per_device_train_batch_size = 32
    local_rank = -1


class CoCoColBERTer(CoColBERTer):
    """
    CoColBERTer model
    """

    config_class = CoCoColBERTerConfig
    base_model_prefix = "bert_model"

    # is_teacher_model = False  # gets overwritten by the dynamic teacher runner

    @staticmethod
    def from_config(config):
        cfg = CoCoColBERTerConfig()
        cfg.bert_model = config["bert_pretrained_model"]
        cfg.return_vecs = config.get("in_batch_negatives", False)
        cfg.dual_loss = config.get("train_dual_loss", False)
        cfg.trainable = config["bert_trainable"]

        cfg.compression_dim = config["colberter_compression_dim"]
        cfg.retrieval_compression_dim = config["colberter_retrieval_compression_dim"]
        cfg.use_contextualized_stopwords = config["colberter_use_contextualized_stopwords"]
        cfg.aggregate_unique_ids = config["colberter_aggregate_unique_ids"]
        cfg.compress_to_exact_mini_mode = config.get("colberter_compress_to_exact_mini_mode", False)
        cfg.second_compress_dim = config.get("colberter_second_compress_dim", -1)

        # condenser parameters
        cfg.co_n_head_layers = config.get("co_n_head_layers", -1)
        cfg.skip_from = config.get("skip_from", 2)

        # cocondenser parameters
        cfg.per_device_train_batch_size = config.get("batch_size_train", 32)
        cfg.local_rank = config.get("cocondenser_local_rank", -1)
        cfg.token_embedding_weight = config.get("token_embedding_weight", -1)

        cfg.cache_chunk_size = config.get("cache_chunk_size", -1)

        return CoCoColBERTer(cfg)


    def __init__(self,
                 cfg) -> None:
        super().__init__(cfg)

        self.return_vecs = cfg.return_vecs
        self.dual_loss = cfg.dual_loss

        self.bert_model = AutoModel.from_pretrained(cfg.bert_model)
        # previously this was AutoModelForMaskedMLM.from_pretrained(cfg.bert_model)
        self.mlm_loss_fct = nn.CrossEntropyLoss()

        # only mlm head in forward integrate!
        self.use_compressor = cfg.compression_dim > -1
        if self.use_compressor:
            self.activation_compr = get_activation(self.bert_model.config.activation)
            self.vocab_transform_compr = nn.Linear(cfg.compression_dim, cfg.compression_dim)
            self.vocab_layer_norm_compr = nn.LayerNorm(cfg.compression_dim, eps=1e-12)
            self.vocab_projector_compr = nn.Linear(cfg.compression_dim, self.bert_model.config.vocab_size)

            self.vocab_transform_compr.apply(self.bert_model._init_weights)
            self.vocab_layer_norm_compr.apply(self.bert_model._init_weights)
            self.vocab_projector_compr.apply(self.bert_model._init_weights)

        self.activation = get_activation(self.bert_model.config.activation)
        self.vocab_transform = nn.Linear(self.bert_model.config.dim, self.bert_model.config.dim)
        self.vocab_layer_norm = nn.LayerNorm(self.bert_model.config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(self.bert_model.config.dim, self.bert_model.config.vocab_size)

        self.vocab_transform.apply(self.bert_model._init_weights)
        self.vocab_layer_norm.apply(self.bert_model._init_weights)
        self.vocab_projector.apply(self.bert_model._init_weights)

        self.vocab_size = self.bert_model.config.vocab_size

        for p in self.bert_model.parameters():
            p.requires_grad = cfg.trainable

        self.use_compressor = cfg.compression_dim > -1
        if self.use_compressor:
            self.compressor = torch.nn.Linear(self.bert_model.config.hidden_size, cfg.compression_dim)
            # need a decompressor for tokens
            self.decompressor = torch.nn.Linear(cfg.compression_dim, self.bert_model.config.hidden_size)

        self.use_retrieval_compression = cfg.retrieval_compression_dim > -1
        if self.use_retrieval_compression:
            self.compressor_retrieval = torch.nn.Linear(self.bert_model.config.hidden_size,
                                                        cfg.retrieval_compression_dim)
            self.decompressor_retrieval = torch.nn.Linear(cfg.retrieval_compression_dim,
                                                          self.bert_model.config.hidden_size)

        self.aggregate_unique_ids = cfg.aggregate_unique_ids

        # inserted from condenser
        self.use_co_n_head_layers = cfg.co_n_head_layers > -1
        if self.use_co_n_head_layers:
            self.c_head = nn.ModuleList(
                [TransformerBlock(self.bert_model.config) for _ in range(cfg.co_n_head_layers)]) #BertLayer
            self.c_head.apply(self.bert_model._init_weights)   #_init_weights()

        self.skip_from = cfg.skip_from

        # for cocondenser training, only for gradient caching
        self.gc = cfg.cache_chunk_size > -1
        if self.gc:
            effective_bsz = cfg.per_device_train_batch_size * self._world_size() #* 2
        else:
            effective_bsz = cfg.per_device_train_batch_size * self._world_size()
        target = torch.arange(effective_bsz, dtype=torch.long).view(-1, 2).flip([1]).flatten().contiguous()

        self.local_rank = cfg.local_rank
        self.token_embedding_weight = cfg.token_embedding_weight
        self.per_device_train_batch_size = cfg.per_device_train_batch_size

        self.register_buffer('co_target', target)

        #self.use_contextualized_stopwords = cfg.use_contextualized_stopwords
        #if self.use_contextualized_stopwords:
        #    self.stop_word_reducer = nn.Linear(cfg.compression_dim, 1, bias=True)
        #    torch.nn.init.constant_(self.stop_word_reducer.bias, 1)  # make sure we don't start in a broken state

        #self.compress_to_exact_mini_mode = cfg.compress_to_exact_mini_mode
        #self.use_second_compression = cfg.second_compress_dim > -1
        #if self.use_second_compression:
        #    self.mini_compressor = torch.nn.Linear(cfg.compression_dim, cfg.second_compress_dim)

    def _gather_tensor(self, t: Tensor):
        all_tensors = [torch.empty_like(t) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(all_tensors, t)
        all_tensors[self.local_rank] = t
        return all_tensors

    def gather_tensors(self, *tt: Tensor):
        tt = [torch.cat(self._gather_tensor(t)) for t in tt]
        return tt


    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                labels: Dict[str, torch.LongTensor],
                # from cocondenser
                grad_cache: Tensor = None,
                chunk_offset: int = None,
                use_fp16: bool = True,
                output_secondary_output: bool = False) -> torch.Tensor:

        with torch.cuda.amp.autocast(enabled=use_fp16):
            retrieval_vec, vecs, mask, output, mlm_loss, co_loss, co_cls_hiddens, surrogate = \
                self.forward_representation(tokens, labels, grad_cache, chunk_offset)


            if self.use_retrieval_compression:
                cls_hiddens = self.decompressor_retrieval(retrieval_vec)
            else:
                cls_hiddens = retrieval_vec

            # maybe i need to include that in the forward_shared_encoding where the maskedlmoutput happens!
            if self.use_compressor:
                vecs = self.decompressor(vecs)

            # cls_hiddens2 is the original implementation from cocondenser but without the compression
            #cls_hiddens2 = lm_out.hidden_states[-1][:, :1]
            #cls_hiddens = cls_hiddens2
            skip_hiddens = output.hidden_states[self.skip_from]

            #print(skip_hiddens)
            #print("skip hiddens nan" if torch.any(torch.isnan(skip_hiddens)) == True else "")

            #cls_hiddens = cls_hiddens.unsqueeze(1)

            hiddens = torch.cat([cls_hiddens, skip_hiddens[:, 1:]], dim=1)

            for layer in self.c_head:
                layer_out = layer(
                    hiddens,
                    mask,
                )
                hiddens = layer_out[0]

            # cls does not exist for distilbert
            #scores = self.bert_model.cls(hiddens)
            # is this score only the output from the pretrainig head?
            #https: // github.com / huggingface / transformers / blob / 34097
            #b3304d79ace845316d4929220623279c8bc / src / transformers / models / bert / modeling_bert.py  # L721

            # distilbertformaskedlm https://github.com/huggingface/transformers/blob/main/src/transformers/models/distilbert/modeling_distilbert.py
            prediction_logits = self.vocab_transform(hiddens)  # (bs, seq_length, dim)
            prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
            scores = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)

            return scores, mlm_loss, co_loss, surrogate


    def forward_representation(self,
                               tokens: Dict[str, torch.LongTensor],
                               labels: Dict[str, torch.LongTensor],
                               grad_cache,
                               chunk_offset) -> torch.Tensor:

        retrieval_vec, vecs, mask, output, mlm_loss, co_loss, co_cls_hiddens, surrogate = \
            self.forward_shared_encoding(tokens, labels, grad_cache, chunk_offset)

        return retrieval_vec, vecs, mask, output, mlm_loss, co_loss, co_cls_hiddens, surrogate


    def forward_shared_encoding(self,  # type: ignore
                                tokens: Dict[str, torch.LongTensor],
                                labels: Dict[str, torch.LongTensor],
                                grad_cache: Dict[str, torch.LongTensor]=None,
                                chunk_offset: int=None):
        #
        # ColBERT-style encoding
        #
        mask = tokens["attention_mask"].bool()
        output = self.bert_model(
                                    #**tokens,
                                 input_ids=tokens["input_ids"],
                                 attention_mask=tokens["attention_mask"],
                                 output_hidden_states=True,
                                 return_dict=True)

        vecs = output[0]

        #
        # ColBERTer: Enhanced Reduction
        # ------------------------------------

        #
        # Part 1: Retrieval, Compress [CLS] vectors only to self.compressor_retrieval dim
        #
        if self.use_retrieval_compression:
            retrieval_vec = self.compressor_retrieval(output.hidden_states[-1][:, :1]) #vecs[:, 0, :]
        else:
            retrieval_vec = output.hidden_states[-1][:, :1]   #vecs[:, 0, :] = output[0][:, 0, :]

        # for cocondenser
        if self.local_rank > -1 and grad_cache is None:
            co_cls_hiddens = self.gather_tensors(retrieval_vec.squeeze().contiguous())[0]
        else:
            co_cls_hiddens = retrieval_vec.squeeze()

        #
        # Part 2, Refinement 2.1: Dimensionality Reduction of Compressed Sub-Word Representations
        # -> seems counter-intuitive to do first, but saves a lot of memory consumption for the aggregation
        #
        if self.use_compressor:
            vecs = self.compressor(vecs)

        #
        # Part 2, Refinement 2.2: Aggregate unique-whole-words
        #
        if self.aggregate_unique_ids:
            #print("use aggregate unique_ids")
            # aggregate whole-words
            aggregation_mask = (tokens["unique_words"].unsqueeze(-1) == tokens["input_ids_to_words_map"].unsqueeze(
                1)).unsqueeze(-1)
            aggregated_vecs = (vecs.unsqueeze(1).expand(-1, aggregation_mask.shape[1], -1, -1) * aggregation_mask).sum(
                2)

            # mean pooling (instead of sum)
            aggregated_vecs = aggregated_vecs / aggregation_mask.float().sum(-2)

            whole_word_mask = tokens["unique_words"] > 0
            #vecs = aggregated_vecs

            # deaggregation
            deaggregated_vecs = (
                        aggregated_vecs.unsqueeze(2).expand(-1, -1, aggregation_mask.shape[2], -1) * aggregation_mask).sum(1)

            if self.token_embedding_weight > -1:
                deaggregated_vecs = deaggregated_vecs + self.token_embedding_weight * vecs

            vecs = deaggregated_vecs

            # and then the decompression step as well?
            # and then the masked language modelling loss?

            # here add token embedding additionally? how can i deaggregate it?

        if self.use_compressor:
            prediction_logits = self.vocab_transform_compr(vecs)  # (bs, seq_length, dim)
            prediction_logits = self.activation_compr(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_layer_norm_compr(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_projector_compr(prediction_logits)
        else:
            # here the mlm loss, so that it takes the right vecs, in case I use compression before or aggregation!
            prediction_logits = self.vocab_transform(vecs)  # (bs, seq_length, dim)
            prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_projector(prediction_logits)

        mlm_loss = None
        if labels is not None:
            # -100 index = padding token
            mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)),
                                         labels["input_ids"].view(-1))

        surrogate = None
        if grad_cache is None:
            co_loss = self.compute_contrastive_loss(co_cls_hiddens)
        else:
            co_loss = mlm_loss * (float(vecs.size(0)) / self.per_device_train_batch_size)
            cached_grads = grad_cache[chunk_offset: chunk_offset + co_cls_hiddens.size(0)]
            surrogate = torch.dot(cached_grads.flatten(), co_cls_hiddens.flatten())

        # here return, back to subtokens
        # then back from compression, to original input i think, then mlm loss? then whole word autoencoding?
        # whole word contrastive learning?

        return retrieval_vec, vecs, mask, output, mlm_loss, co_loss, co_cls_hiddens, surrogate

    @staticmethod
    def _world_size():
        if torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        else:
            return 1

    def compute_contrastive_loss(self, co_cls_hiddens):
        similarities = torch.matmul(co_cls_hiddens, co_cls_hiddens.transpose(0, 1))
        similarities.fill_diagonal_(float('-inf'))
        co_loss = nn.functional.cross_entropy(similarities, self.co_target) * self._world_size()
        return co_loss