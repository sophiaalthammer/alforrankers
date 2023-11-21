from typing import Dict, Union

import torch
from torch import nn as nn

from transformers import AutoModel,DistilBertModel


class ColBERT_Gumbel(nn.Module):
    """
    ColBERT model from: https://arxiv.org/pdf/2004.12832.pdf
    """

    @staticmethod
    def from_config(config):
        return ColBERT_Gumbel(bert_model = config["bert_pretrained_model"],
                       compression_dim=config["colbert_compression_dim"],
                       trainable=config["bert_trainable"])
    
    def __init__(self,
                 bert_model: Union[str, AutoModel],
                 compression_dim: int = 128,
                 document_topk: int = 25,
                 dropout: float = 0.0,
                 trainable: bool = True) -> None:
        super().__init__()

        if isinstance(bert_model, str):
            self.bert_model = AutoModel.from_pretrained(bert_model)
        else:
            self.bert_model = bert_model

        for p in self.bert_model.parameters():
            p.requires_grad = trainable

        self._dropout = torch.nn.Dropout(p=dropout)
        self.compressor = torch.nn.Linear(self.bert_model.config.hidden_size, compression_dim)
        torch.nn.init.uniform_(self.compressor.weight, -0.001, 0.001)

        self.sampler = GumbelSampler(self.bert_model.config.hidden_size,document_topk)
        self.gumbel_repeat = 10

    def forward(self,
                query: Dict[str, torch.LongTensor],
                document: Dict[str, torch.LongTensor],
                output_secondary_output: bool = False) -> Dict[str, torch.Tensor]:

        #query_vecs = torch.nn.functional.normalize(self.forward_representation(query), p=2, dim=-1)
        #document_vecs = torch.nn.functional.normalize(self.forward_representation(document), p=2, dim=-1)
        
        query_vecs = self.forward_representation(query,"query")
        document_vecs = self.forward_representation(document,"doc")



        if self.training:
            score = torch.zeros((query_vecs.shape[0]),device=query_vecs.device,requires_grad=True)
            for _ in range(self.gumbel_repeat):

                sampled_doc_vecs,document_mask,sampled_ind = self.sampler(document_vecs, document["tokens"]["mask"])

                score_samp = torch.bmm(query_vecs, sampled_doc_vecs.transpose(2,1))
                score_samp[~document_mask.unsqueeze(1).expand(-1,score_samp.shape[1],-1)] = - 10000

                score_samp = score_samp.max(-1).values

                score_samp[~query["tokens"]["mask"]] = 0

                score = score + score_samp.sum(-1)

            score = score / self.gumbel_repeat
        else:
            sampled_doc_vecs,document_mask,sampled_ind = self.sampler(document_vecs, document["tokens"]["mask"])
            score = torch.bmm(query_vecs, sampled_doc_vecs.transpose(2,1))
            score[~document_mask.unsqueeze(1).expand(-1,score.shape[1],-1)] = - 10000
            score = score.max(-1).values
            score[~query["tokens"]["mask"]] = 0
            score = score.sum(-1) 


        if output_secondary_output:
            return score, {}
        return score

    def forward_representation(self,  # type: ignore
                               tokens: Dict[str, torch.LongTensor],
                               sequence_type:str) -> Dict[str, torch.Tensor]:
        mask = tokens["tokens"]["mask"]
        if self.bert_model.base_model_prefix == 'distilbert': # diff input / output 
            vecs = self.bert_model(input_ids=tokens["tokens"]["token_ids"],
                                     attention_mask=mask)[0]
        elif self.bert_model.base_model_prefix == 'longformer':
            vecs, _ = self.bert_model(input_ids=tokens["tokens"]["token_ids"],
                                        attention_mask=mask.long(),
                                        global_attention_mask = ((1-tokens["tokens"]["type_ids"])*mask).long())
        elif self.bert_model.base_model_prefix == 'roberta': # no token type ids
            vecs, _ = self.bert_model(input_ids=tokens["tokens"]["token_ids"],
                                        attention_mask=mask)
        else:
            vecs, _ = self.bert_model(input_ids=tokens["tokens"]["token_ids"],
                                        token_type_ids=tokens["tokens"]["type_ids"],
                                        attention_mask=mask)

        vecs = self.compressor(vecs)

        return vecs

    def get_param_stats(self):
        return "BERT_cls: / "
    def get_param_secondary(self):
        return {}

#
# Gumbel Sampler is a new module, with new parameters added to the existing bert module
# -> forward pass: compute logits for gumbel based on representations & return gumbel-sampled tensors
# -> backward pass: gradient flow to the logit component is possible due to gumbel-trick & merging
#
class GumbelSampler(nn.Module):

    def __init__(self, rep_dim,sample_top_k):
        super().__init__()
        self.logit_reducer = nn.Linear(rep_dim, rep_dim // 2, bias=True)
        self.logit_reducer2 = nn.Linear(rep_dim // 2, 1, bias=True)
        #torch.nn.init.constant_(self.logit_reducer2.bias, 2)

        self.sample_top_k=sample_top_k

    def forward(self, reps, mask, probs=None):
        if probs is not None:
            logits = probs
        else:
            #logits = torch.log(torch.clamp(self.logit_reducer2(torch.tanh(self.logit_reducer(reps))),0.0001)).squeeze(-1)
            #logits = torch.tanh(self.logit_reducer2(torch.tanh(self.logit_reducer(reps)))).squeeze(-1)
            logits = self.logit_reducer2(torch.tanh(self.logit_reducer(reps))).squeeze(-1)

        # shape same as reps, data: 1/0 to sample or not to sample 
        gumbel_mask, gumbel_indices = self.gumbel_softmax(logits = logits, 
                                                         mask = (~mask*-10000),
                                                         topk = self.sample_top_k,
                                                         temperature = 0.1)

        # -> needs to be multiplied for gradient flow (just indexing does only take the gradient from the slice, but not the indexing var)
        # now, logit_reducer's get grads after .backward()
        #
        # sampled_reps & mask now have sequence length of topk
        if self.training: # not sure if keeping the empty slices is actually needed (could be for the gradient flow to those indices?)
            sampled_reps = reps * gumbel_mask.unsqueeze(-1)
            sampled_mask = mask
            sampled_mask[gumbel_mask == 0] = False
        else: # reduce the size for inference
            sampled_reps = reps[gumbel_mask.bool(),:].view(reps.shape[0],-1,reps.shape[-1])
            sampled_mask = mask[gumbel_mask.bool()].view(reps.shape[0],-1)

        return sampled_reps, sampled_mask, gumbel_indices

    # gumbel code from: https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
    def sample_gumbel(self, shape,device, eps=1e-20):
        U = torch.rand(shape,device=device)
        return - torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, mask, temperature):

        #rand baseline
        #y = self.sample_gumbel(logits.size(),logits.device)

        # todo: does this make sense? -> only add noise during training, keep inference deterministic
        if self.training:
            gumbel_noise = self.sample_gumbel(logits.size(), logits.device)
            y = logits + gumbel_noise
        else:
            y = logits

        # add mask (-10000 entries for padding) to get a masked softmax
        # -> discourage sampling from padding
        y = y + mask
        return torch.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, mask, topk, temperature):
        """

        applies the gumbel-softmax once and then takes the  and takes the argmax each time, masking previous picks

        input: [*, n_class]
        return: [*, n_class] boolean sample mask, indices selected
        """
        y = self.gumbel_softmax_sample(logits, mask, temperature)

        #
        # special first index cls vector handling, force to sample the first vector
        # could be removed for tasks not depending on cls
        # added here to not mess with the softmax
        #
        cls_force = torch.zeros_like(y)
        cls_force[:,0] = 10000
        y = y + cls_force

        shape = y.size()
        _, ind = y.topk(topk, dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind, 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y , ind

    def gumbel_k_times_softmax(self, logits, mask, topk, temperature):
        """
        
        applies the gumbel-softmax k-times and takes the argmax each time, masking previous picks

        input: [*, n_class]
        return: [*, n_class] boolean sample mask, indices selected
        """
        #
        # special first index cls vector handling, force to sample the first vector
        # could be removed for tasks not depending on cls
        #
        logits[:,0] = 100 + logits.max(-1)[0]
        top_k_mask = mask.clone()

        y_hard = torch.zeros_like(logits)
        y = torch.zeros_like(logits)
        all_ind = []
        for i in range(topk):
            y += self.gumbel_softmax_sample(logits, top_k_mask, temperature)
            y[y_hard == 1] = 0

            ind = y.argmax(dim=-1)
            all_ind.append(ind.unsqueeze(-1))
            y_hard.scatter_(1, ind.unsqueeze(-1), 1)
            y = (y_hard - y).detach() + y

            top_k_mask[y_hard == 1] = -10000

        return y, torch.cat(all_ind,dim=1)

class ColBERT_Gumbel2(nn.Module):
    """
    ColBERT model from: https://arxiv.org/pdf/2004.12832.pdf
    """

    @staticmethod
    def from_config(config):
        return ColBERT_Gumbel2(bert_model = config["bert_pretrained_model"],
                       compression_dim=config["colbert_compression_dim"],
                       trainable=config["bert_trainable"])
    
    def __init__(self,
                 bert_model: Union[str, AutoModel],
                 compression_dim: int = 768,
                 document_topk: int = 25,
                 dropout: float = 0.0,
                 trainable: bool = True) -> None:
        super().__init__()

        if isinstance(bert_model, str):
            self.bert_model = AutoModel.from_pretrained(bert_model)
        else:
            self.bert_model = bert_model

        for p in self.bert_model.parameters():
            p.requires_grad = trainable

        self._dropout = torch.nn.Dropout(p=dropout)
        self.compressor = torch.nn.Linear(self.bert_model.config.hidden_size, compression_dim)
        torch.nn.init.uniform_(self.compressor.weight, -0.001, 0.001)

        self.sampler = GumbelSampler(self.bert_model.config.hidden_size,document_topk)

    def forward(self,
                query: Dict[str, torch.LongTensor],
                document: Dict[str, torch.LongTensor],
                output_secondary_output: bool = False) -> Dict[str, torch.Tensor]:

        #query_vecs = torch.nn.functional.normalize(self.forward_representation(query), p=2, dim=-1)
        #document_vecs = torch.nn.functional.normalize(self.forward_representation(document), p=2, dim=-1)
        
        query_vecs,_ = self.forward_representation(query,"query")
        document_vecs,document_mask = self.forward_representation(document,"doc")


        score = torch.bmm(query_vecs, document_vecs.transpose(2,1))
        #score[~query["tokens"]["mask"].unsqueeze(-1).expand(-1,-1,score.shape[-1])] = - 10000
        score[~document_mask.unsqueeze(1).expand(-1,score.shape[1],-1)] = - 10000
        
        score = score.max(-1).values
        
        score[~query["tokens"]["mask"]] = 0
        
        score = score.sum(-1)

        if output_secondary_output:
            return score, {}
        return score

    def forward_representation(self,  # type: ignore
                               tokens: Dict[str, torch.LongTensor],
                               sequence_type:str) -> Dict[str, torch.Tensor]:
        #if self.bert_model.base_model_prefix == 'distilbert': # diff input / output 
        vecs, mask = self.bert_model(input_ids=tokens["tokens"]["token_ids"],
                               attention_mask=tokens["tokens"]["mask"],
                               sequence_type=sequence_type)
        #elif self.bert_model.base_model_prefix == 'longformer':
        #    vecs, _ = self.bert_model(input_ids=tokens["tokens"]["token_ids"],
        #                                attention_mask=mask.long(),
        #                                global_attention_mask = ((1-tokens["tokens"]["type_ids"])*mask).long())
        #elif self.bert_model.base_model_prefix == 'roberta': # no token type ids
        #    vecs, _ = self.bert_model(input_ids=tokens["tokens"]["token_ids"],
        #                                attention_mask=mask)
        #else:
        #    vecs, _ = self.bert_model(input_ids=tokens["tokens"]["token_ids"],
        #                                token_type_ids=tokens["tokens"]["type_ids"],
        #                                attention_mask=mask)
        #if sequence_type == "doc": 
        #    vecs,mask,sampled_ind = self.sampler(vecs, mask)

        vecs = self.compressor(vecs)

        return vecs, mask

    def get_param_stats(self):
        return "BERT_cls: / "
    def get_param_secondary(self):
        return {}

#
# Bert split changes the distilbert model from huggingface to bea able to split query and document until a set layer
#
class Bert_stopword_Gumbel(DistilBertModel):

    @staticmethod
    def from_config(config):
        return Bert_stopword_Gumbel(config=DistilBertConfig.from_pretrained(config["bert_pretrained_model"]),
                          join_layer_idx=3)

    def __init__(self, config, join_layer_idx):
        super().__init__(config)
        self.transformer = StopwordTransformer(config,join_layer_idx)  # Encoder


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        sequence_type=None
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.DistilBertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)  # (bs, seq_length, dim)

        output_vecs, output_mask = self.transformer(
            x=inputs_embeds,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            sequence_type=sequence_type
        )

        return output_vecs,output_mask  


class StopwordTransformer(nn.Module):
    def __init__(self, config,join_layer_idx):
        super().__init__()
        self.n_layers = config.n_layers

        layer = TransformerBlock(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.n_layers)])
        self.gumbel_sampler = GumbelSampler2(config.hidden_size, 25)
        self.logit_reducer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.join_layer_idx = join_layer_idx

        self.stopword_embeddings = nn.Embedding(2, config.hidden_size)
        self.stopword_embeddings.weight.data = self.stopword_embeddings.weight.data / 10
        self.stop_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, x, attn_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False,sequence_type=None):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
            Input sequence embedded.
        attn_mask: torch.tensor(bs, seq_length)
            Attention mask on the sequence.

        Outputs
        -------
        hidden_state: torch.tensor(bs, seq_length, dim)
            Sequence of hiddens states in the last (top) layer
        all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if output_hidden_states=True
        all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if output_attentions=True
        """

        hidden_state = x
        gumbel_mask = None
        for i, layer_module in enumerate(self.layer):

            layer_outputs = layer_module(
                x=hidden_state, attn_mask=attn_mask, head_mask=head_mask[i], output_attentions=output_attentions
            )
            hidden_state = layer_outputs[-1]

            if i == self.join_layer_idx:
                if sequence_type == "doc":
                    gumbel_mask, gumbel_indices = self.gumbel_sampler(hidden_state, attn_mask)
                else:
                    gumbel_mask = torch.ones_like(attn_mask)

                hidden_state = self.stop_norm(hidden_state + self.stopword_embeddings(gumbel_mask.long()))

        if self.training: # not sure if keeping the empty slices is actually needed (could be for the gradient flow to those indices?)
            hidden_state = hidden_state * gumbel_mask.unsqueeze(-1)
            attn_mask[gumbel_mask == 0] = False
        else: # reduce the size for inference
            hidden_state = hidden_state[gumbel_mask.bool(),:].view(hidden_state.shape[0],-1,hidden_state.shape[-1])
            attn_mask = attn_mask[gumbel_mask.bool()].view(hidden_state.shape[0],-1)
   
        return hidden_state, attn_mask 


    #def forward(self, query_embs, query_mask, doc_embs, doc_mask,join_layer_idx, output_attentions=False, output_hidden_states=False):
    #    """
    #    Parameters
    #    ----------
    #    x: torch.tensor(bs, seq_length, dim)
    #        Input sequence embedded.
    #    attn_mask: torch.tensor(bs, seq_length)
    #        Attention mask on the sequence.
#
    #    Outputs
    #    -------
    #    hidden_state: torch.tensor(bs, seq_length, dim)
    #        Sequence of hiddens states in the last (top) layer
    #    all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
    #        Tuple of length n_layers with the hidden states from each layer.
    #        Optional: only if output_hidden_states=True
    #    all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
    #        Tuple of length n_layers with the attention weights from each layer
    #        Optional: only if output_attentions=True
    #    """
    #    all_hidden_states = ()
    #    all_attentions = ()
#
    #    #
    #    # query / doc sep.
    #    #
    #    hidden_state_q = query_embs
    #    hidden_state_d = doc_embs
    #    for layer_module in self.layer[:join_layer_idx]:
#
    #        layer_outputs_q = layer_module(
    #            x=hidden_state_q, attn_mask=query_mask, head_mask=None, output_attentions=output_attentions
    #        )
    #        hidden_state_q = layer_outputs_q[-1]
#
    #        layer_outputs_d = layer_module(
    #            x=hidden_state_d, attn_mask=doc_mask, head_mask=None, output_attentions=output_attentions
    #        )
    #        hidden_state_d = layer_outputs_d[-1]
#
    #    #
    #    # Gumbel !
    #    #
    #    probs = torch.bmm(self.logit_reducer(hidden_state_q[:,0,:].unsqueeze(1)), self.logit_reducer(hidden_state_d).transpose(2,1)).squeeze(1)
#
    #    #score[~query["tokens"]["mask"].unsqueeze(-1).expand(-1,-1,score.shape[-1])] = - 10000
    #    #probs[~doc_mask.unsqueeze(1).expand(-1,probs.shape[1],-1)] = - 10000
#
    #    hidden_state_d,doc_mask,sampled_ind = self.gumbel_sampler(hidden_state_d, probs, doc_mask)
#
    #    #
    #    # combine
    #    #
    #    x = torch.cat([hidden_state_q,hidden_state_d],dim=1)
    #    attn_mask = torch.cat([query_mask,doc_mask],dim=1)
#
    #    #
    #    # combined
    #    #
    #    hidden_state = x
    #    for layer_module in self.layer[join_layer_idx:]:
    #        layer_outputs = layer_module(
    #            x=hidden_state, attn_mask=attn_mask, head_mask=None, output_attentions=output_attentions
    #        )
    #        hidden_state = layer_outputs[-1]
#
    #    # Add last layer
    #    if output_hidden_states:
    #        all_hidden_states = all_hidden_states + (hidden_state,)
#
    #    outputs = (hidden_state,)
    #    if output_hidden_states:
    #        outputs = outputs + (all_hidden_states,)
    #    if output_attentions:
    #        outputs = outputs + (all_attentions,)
    #    return outputs  # last-layer hidden state, (all hidden states), (all attentions)



#
# Gumbel Sampler is a new module, with new parameters added to the existing bert module
# -> forward pass: compute logits for gumbel based on representations & return gumbel-sampled tensors
# -> backward pass: gradient flow to the logit component is possible due to gumbel-trick & merging
#
class GumbelSampler2(nn.Module):

    def __init__(self, rep_dim,sample_top_k):
        super().__init__()
        self.logit_reducer = nn.Linear(rep_dim, rep_dim // 2, bias=True)
        self.logit_reducer2 = nn.Linear(rep_dim // 2, 1, bias=True)
        torch.nn.init.constant_(self.logit_reducer2.bias, 2)

        self.sample_top_k=sample_top_k

    def forward(self, reps, mask, probs=None):
        if probs is not None:
            logits = probs
        else:
            logits = torch.log(torch.clamp(self.logit_reducer2(torch.tanh(self.logit_reducer(reps))),0.0001)).squeeze(-1)
            #logits = torch.tanh(self.logit_reducer2(torch.tanh(self.logit_reducer(reps)))).squeeze(-1)
            #logits = self.logit_reducer2(torch.tanh(self.logit_reducer(reps))).squeeze(-1)

        # shape same as reps, data: 1/0 to sample or not to sample 
        gumbel_mask, gumbel_indices = self.gumbel_softmax(logits = logits, 
                                                         mask = (~mask*-10000),
                                                         topk = self.sample_top_k,
                                                         temperature = 1)

        # -> needs to be multiplied for gradient flow (just indexing does only take the gradient from the slice, but not the indexing var)
        # now, logit_reducer's get grads after .backward()
        #
        # sampled_reps & mask now have sequence length of topk
        #if self.training: # not sure if keeping the empty slices is actually needed (could be for the gradient flow to those indices?)
        #    sampled_reps = reps * gumbel_mask.unsqueeze(-1)
        #    sampled_mask = mask
        #    sampled_mask[gumbel_mask == 0] = False
        #else: # reduce the size for inference
        #    sampled_reps = reps[gumbel_mask.bool(),:].view(reps.shape[0],-1,reps.shape[-1])
        #    sampled_mask = mask[gumbel_mask.bool()].view(reps.shape[0],-1)

        return gumbel_mask, gumbel_indices

    # gumbel code from: https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
    def sample_gumbel(self, shape,device, eps=1e-20):
        U = torch.rand(shape,device=device)
        return - torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, mask, temperature):

        #rand baseline
        #y = self.sample_gumbel(logits.size(),logits.device)

        # todo: does this make sense? -> only add noise during training, keep inference deterministic
        if self.training:
            gumbel_noise = self.sample_gumbel(logits.size(), logits.device)
            y = logits + gumbel_noise
        else:
            y = logits

        # add mask (-10000 entries for padding) to get a masked softmax
        # -> discourage sampling from padding
        y = y + mask
        return torch.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, mask, topk, temperature):
        """

        applies the gumbel-softmax once and then takes the  and takes the argmax each time, masking previous picks

        input: [*, n_class]
        return: [*, n_class] boolean sample mask, indices selected
        """
        y = self.gumbel_softmax_sample(logits, mask, temperature)

        #
        # special first index cls vector handling, force to sample the first vector
        # could be removed for tasks not depending on cls
        # added here to not mess with the softmax
        #
        cls_force = torch.zeros_like(y)
        cls_force[:,0] = 10000
        y = y + cls_force

        shape = y.size()
        _, ind = y.topk(topk, dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind, 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y , ind

    def gumbel_k_times_softmax(self, logits, mask, topk, temperature):
        """
        
        applies the gumbel-softmax k-times and takes the argmax each time, masking previous picks

        input: [*, n_class]
        return: [*, n_class] boolean sample mask, indices selected
        """
        #
        # special first index cls vector handling, force to sample the first vector
        # could be removed for tasks not depending on cls
        #
        logits[:,0] = 100 + logits.max(-1)[0]
        top_k_mask = mask.clone()

        y_hard = torch.zeros_like(logits)
        y = torch.zeros_like(logits)
        all_ind = []
        for i in range(topk):
            y += self.gumbel_softmax_sample(logits, top_k_mask, temperature)
            y[y_hard == 1] = 0

            ind = y.argmax(dim=-1)
            all_ind.append(ind.unsqueeze(-1))
            y_hard.scatter_(1, ind.unsqueeze(-1), 1)
            y = (y_hard - y).detach() + y

            top_k_mask[y_hard == 1] = -10000

        return y, torch.cat(all_ind,dim=1)
