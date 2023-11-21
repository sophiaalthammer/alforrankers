from transformers import *
import math
import torch
from torch import nn as nn

#
# Bert split changes the distilbert model from huggingface to bea able to split query and document until a set layer
#
class PreTTR_Gumbel(DistilBertModel):

    @staticmethod
    def from_config(config):
        return PreTTR_Gumbel(config=DistilBertConfig.from_pretrained(config["bert_pretrained_model"]),
                          join_layer_idx=3)

    def __init__(self, config, join_layer_idx):
        super().__init__(config)
        self.transformer = SplitTransformer(config)  # Encoder
        self.embeddings = PosOffsetEmbeddings(config)  # Embeddings
        self._classification_layer = torch.nn.Linear(self.config.hidden_size, 1,bias=False)

        self.join_layer_idx = join_layer_idx

    def forward(
        self,
        query,
        document,
        output_secondary_output=False):

        query_input_ids=query["tokens"]["token_ids"]
        query_attention_mask=query["tokens"]["mask"]

        document_input_ids=document["tokens"]["token_ids"][:,1:]
        document_attention_mask=document["tokens"]["mask"][:,1:]


        query_embs = self.embeddings(query_input_ids)  # (bs, seq_length, dim)
        document_embs = self.embeddings(document_input_ids,query_input_ids.shape[-1])  # (bs, seq_length, dim)

        tfmr_output = self.transformer(
            query_embs=query_embs, 
            query_mask=query_attention_mask, 
            doc_embs=document_embs, 
            doc_mask=document_attention_mask,
            join_layer_idx = self.join_layer_idx
        )
        hidden_state = tfmr_output[0]

        score = self._classification_layer(hidden_state[:,0,:]).squeeze()

        if output_secondary_output:
            return score, {}
        return score

    def get_param_stats(self):
        return "PreTTR: / "
    def get_param_secondary(self):
        return {}

class PosOffsetEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=config.max_position_embeddings, dim=config.dim, out=self.position_embeddings.weight
            )

        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, pos_offset=0):
        """
        Parameters
        ----------
        input_ids: torch.tensor(bs, max_seq_length)
            The token ids to embed.

        Outputs
        -------
        embeddings: torch.tensor(bs, max_seq_length, dim)
            The embedded tokens (plus position embeddings, no token_type embeddings)
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids) + pos_offset  # (bs, max_seq_length)

        word_embeddings = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)
        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        embeddings = word_embeddings + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings

class SplitTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layers = config.n_layers

        layer = TransformerBlock(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.n_layers)])
        self.gumbel_sampler = GumbelSampler(config.hidden_size, 25)
        self.logit_reducer = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, query_embs, query_mask, doc_embs, doc_mask,join_layer_idx, output_attentions=False, output_hidden_states=False):
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
        all_hidden_states = ()
        all_attentions = ()

        #
        # query / doc sep.
        #
        hidden_state_q = query_embs
        hidden_state_d = doc_embs
        for layer_module in self.layer[:join_layer_idx]:

            layer_outputs_q = layer_module(
                x=hidden_state_q, attn_mask=query_mask, head_mask=None, output_attentions=output_attentions
            )
            hidden_state_q = layer_outputs_q[-1]

            layer_outputs_d = layer_module(
                x=hidden_state_d, attn_mask=doc_mask, head_mask=None, output_attentions=output_attentions
            )
            hidden_state_d = layer_outputs_d[-1]

        #
        # Gumbel !
        #
        probs = torch.bmm(self.logit_reducer(hidden_state_q[:,0,:].unsqueeze(1)), self.logit_reducer(hidden_state_d).transpose(2,1)).squeeze(1)

        #score[~query["tokens"]["mask"].unsqueeze(-1).expand(-1,-1,score.shape[-1])] = - 10000
        #probs[~doc_mask.unsqueeze(1).expand(-1,probs.shape[1],-1)] = - 10000

        hidden_state_d,doc_mask,sampled_ind = self.gumbel_sampler(hidden_state_d, probs, doc_mask)

        #
        # combine
        #
        x = torch.cat([hidden_state_q,hidden_state_d],dim=1)
        attn_mask = torch.cat([query_mask,doc_mask],dim=1)

        #
        # combined
        #
        hidden_state = x
        for layer_module in self.layer[join_layer_idx:]:
            layer_outputs = layer_module(
                x=hidden_state, attn_mask=attn_mask, head_mask=None, output_attentions=output_attentions
            )
            hidden_state = layer_outputs[-1]

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        outputs = (hidden_state,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

#
# Gumbel Sampler is a new module, with new parameters added to the existing bert module
# -> forward pass: compute logits for gumbel based on representations & return gumbel-sampled tensors
# -> backward pass: gradient flow to the logit component is possible due to gumbel-trick & merging
#
class GumbelSampler(nn.Module):

    def __init__(self, rep_dim,sample_top_k):
        super().__init__()
        #self.logit_reducer = nn.Linear(rep_dim, rep_dim // 2, bias=True)
        #self.logit_reducer2 = nn.Linear(rep_dim // 2, 1, bias=True)
        #torch.nn.init.constant_(self.logit_reducer2.bias, 2)

        self.sample_top_k=sample_top_k

    def forward(self, reps, probs, mask):
        #if probs is not None:
        logits = probs
        #else:
        #    logits = torch.log(torch.clamp(self.logit_reducer2(torch.tanh(self.logit_reducer(reps))),0.0001)).squeeze(-1)
        #    #logits = torch.tanh(self.logit_reducer2(torch.tanh(self.logit_reducer(reps)))).squeeze(-1)
        #    #logits = self.logit_reducer2(torch.tanh(self.logit_reducer(reps))).squeeze(-1)
#
        # shape same as reps, data: 1/0 to sample or not to sample 
        gumbel_mask, gumbel_indices = self.gumbel_k_times_softmax(logits = logits, 
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
        cls_force[:,0] = 1
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
