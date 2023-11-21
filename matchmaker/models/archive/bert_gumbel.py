#from transformers import *
#import math
#import torch
#from torch import nn as nn
#
##
## Bert gumbel augments & changes a huggingface transformers bert model 
## -> sub-sample 
##
#class Bert_Gumbel(BertModel):
#
#    def __init__(self, config):
#        super().__init__(config)
#        #self.encoder = BertGumbelEncoder(config)
#        self.gumbel = GumbelSampler(config)
#        self.gumbel_probs = nn.Embedding(config.vocab_size,1,0)
#        torch.nn.init.constant_(self.gumbel_probs.weight, 1)
#
#    def forward(
#        self,
#        input_ids=None,
#        attention_mask=None,
#        token_type_ids=None,
#        position_ids=None,
#        head_mask=None,
#        inputs_embeds=None,
#        encoder_hidden_states=None,
#        encoder_attention_mask=None,
#        output_attentions=None,
#        output_hidden_states=None,
#    ):
#        r"""
#    Return:
#        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
#        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
#            Sequence of hidden-states at the output of the last layer of the model.
#        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
#            Last layer hidden-state of the first token of the sequence (classification token)
#            further processed by a Linear layer and a Tanh activation function. The Linear
#            layer weights are trained from the next sentence prediction (classification)
#            objective during pre-training.
#
#            This output is usually *not* a good summary
#            of the semantic content of the input, you're often better with averaging or pooling
#            the sequence of hidden-states for the whole input sequence.
#        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
#            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
#            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
#
#            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
#        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
#            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
#            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
#
#            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
#            heads.
#        """
#        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#        output_hidden_states = (
#            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#        )
#
#        if input_ids is not None and inputs_embeds is not None:
#            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#        elif input_ids is not None:
#            input_shape = input_ids.size()
#        elif inputs_embeds is not None:
#            input_shape = inputs_embeds.size()[:-1]
#        else:
#            raise ValueError("You have to specify either input_ids or inputs_embeds")
#
#        device = input_ids.device if input_ids is not None else inputs_embeds.device
#
#        if attention_mask is None:
#            attention_mask = torch.ones(input_shape, device=device)
#        if token_type_ids is None:
#            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
#
#        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
#        # ourselves in which case we just need to make it broadcastable to all heads.
#        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
#
#        # If a 2D ou 3D attention mask is provided for the cross-attention
#        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
#        if self.config.is_decoder and encoder_hidden_states is not None:
#            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
#            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
#            if encoder_attention_mask is None:
#                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
#            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
#        else:
#            encoder_extended_attention_mask = None
#
#        # Prepare head mask if needed
#        # 1.0 in head_mask indicate we keep the head
#        # attention_probs has shape bsz x n_heads x N x N
#        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
#        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
#        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
#
#        embedding_output = self.embeddings(
#            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
#        )
#
#        #
#        # gumbel addition
#        #
#        all_sp_vecs=[]
#        do_gumbel = embedding_output.shape[1] > 30
#        if do_gumbel:
#            gumbel_probs = self.gumbel_probs(input_ids).squeeze(-1)
#            embedding_output, extended_attention_mask, gumbel_indices = self.gumbel(embedding_output,extended_attention_mask,gumbel_probs)
#            all_sp_vecs.append(gumbel_indices.detach())
#
#
#        encoder_outputs = self.encoder(
#            embedding_output,
#            attention_mask=extended_attention_mask,
#            head_mask=head_mask,
#            encoder_hidden_states=encoder_hidden_states,
#            encoder_attention_mask=encoder_extended_attention_mask,
#            output_attentions=output_attentions,
#            output_hidden_states=output_hidden_states,
#        )
#        sequence_output = encoder_outputs[0]
#        pooled_output = self.pooler(sequence_output)
#
#        outputs = (sequence_output, pooled_output,) + encoder_outputs[
#            1:
#        ]  # add hidden_states and attentions if they are here
#        return outputs + (all_sp_vecs,)  # sequence_output, pooled_output, (hidden_states), (attentions)
#
#
#class BertGumbelEncoder(BertEncoder):
#    def __init__(self, config):
#        super().__init__(config)
#        self.gumbel = GumbelSampler(config)
#        #self.gumbel2 = GumbelSampler(config)
#        #self.gumbel3 = GumbelSampler(config)
#
#    def forward(
#        self,
#        hidden_states,
#        attention_mask=None,
#        head_mask=None,
#        encoder_hidden_states=None,
#        encoder_attention_mask=None,
#        output_attentions=False,
#        output_hidden_states=False,
#    ):
#        all_hidden_states = ()
#        all_attentions = ()
#        all_sp_vecs = []
#        all_sp_stats = torch.zeros((len(self.layer), 1),device=hidden_states.device,dtype=hidden_states.dtype)
#        
#        do_gumbel = hidden_states.shape[1] > 30
#        if do_gumbel:
#            hidden_states, attention_mask,gumbel_indices = self.gumbel(hidden_states,attention_mask)
#            all_sp_vecs.append(gumbel_indices.detach())
#
#        for i, layer_module in enumerate(self.layer):
#            if output_hidden_states:
#                all_hidden_states = all_hidden_states + (hidden_states,)
#
#            if getattr(self.config, "gradient_checkpointing", False):
#
#                def create_custom_forward(module):
#                    def custom_forward(*inputs):
#                        return module(*inputs, output_attentions)
#
#                    return custom_forward
#
#                layer_outputs = torch.utils.checkpoint.checkpoint(
#                    create_custom_forward(layer_module),
#                    hidden_states,
#                    attention_mask,
#                    head_mask[i],
#                    encoder_hidden_states,
#                    encoder_attention_mask,
#                )
#            else:
#                layer_outputs = layer_module(
#                    hidden_states,
#                    attention_mask,
#                    head_mask[i],
#                    encoder_hidden_states,
#                    encoder_attention_mask,
#                    output_attentions,
#                )
#            hidden_states = layer_outputs[0]
#
#            #
#            # Apply gumbel sampling
#            #
#            #if i == 2 and do_gumbel:
#            #    hidden_states, attention_mask,gumbel_indices = self.gumbel(hidden_states,attention_mask)
#
#            #if i == 6 and do_gumbel:
#            #    hidden_states, attention_mask,gumbel_indices = self.gumbel2(hidden_states,attention_mask)
#
#            #    all_sp_vecs.append(gumbel_indices.detach())
#
#            #if i == 8 and do_gumbel:
#            #    hidden_states, attention_mask = self.gumbel3(hidden_states,attention_mask)
#
#            if output_attentions:
#                all_attentions = all_attentions + (layer_outputs[1],)
#
#        # Add last layer
#        if output_hidden_states:
#            all_hidden_states = all_hidden_states + (hidden_states,)
#
#        outputs = (hidden_states,)
#        if output_hidden_states:
#            outputs = outputs + (all_hidden_states,)
#        if output_attentions:
#            outputs = outputs + (all_attentions,)
#        return outputs + (all_sp_vecs,) # last-layer hidden state, (all hidden states), (all attentions)
#
#    def pack_tensor(self,tensor,mask,sparsity):
#        non_empty = (sparsity != 0).squeeze(-1) # todo what about padding?
#
#        all_lens = non_empty.sum(-1)
#        new_max_len = torch.max(all_lens)
#        new_indices = (torch.arange(new_max_len,device=tensor.device).unsqueeze(-1) < all_lens).t()
#
#        new_tensor = torch.zeros((tensor.shape[0],new_max_len,tensor.shape[-1]),device=tensor.device,dtype=tensor.dtype)
#        new_mask = torch.zeros((tensor.shape[0],new_max_len),device=tensor.device,dtype=tensor.dtype)
#        new_mask += -10000
#
#        new_tensor[new_indices,:] = tensor[non_empty,:]
#        new_mask[new_indices] = mask[non_empty.unsqueeze(1).unsqueeze(1)]
#
#        return new_tensor,new_mask.unsqueeze(1).unsqueeze(1)
#
##
## Gumbel Sampler is a new module, with new parameters added to the existing bert module
## -> forward pass: compute logits for gumbel based on representations & return gumbel-sampled tensors
## -> backward pass: gradient flow to the logit component is possible due to gumbel-trick & merging
##
#class GumbelSampler(nn.Module):
#
#    def __init__(self, config):
#        super().__init__()
#        self.logit_reducer = nn.Linear(config.hidden_size, config.hidden_size // 2, bias=True)
#        self.logit_reducer2 = nn.Linear(config.hidden_size // 2, 1, bias=True)
#        torch.nn.init.constant_(self.logit_reducer2.bias, 2) 
#
#    def forward(self, reps,mask,probs=None):
#        if probs is not None:
#            logits = probs
#        else:
#            logits = torch.log(torch.clamp(self.logit_reducer2(torch.tanh(self.logit_reducer(reps))),0.0001)).squeeze(-1)
#            #logits = torch.tanh(self.logit_reducer2(torch.tanh(self.logit_reducer(reps)))).squeeze(-1)
#            #logits = self.logit_reducer2(torch.tanh(self.logit_reducer(reps))).squeeze(-1)
#
#        # shape same as reps, data: 1/0 to sample or not to sample 
#        gumbel_mask, gumbel_indices = self.gumbel_k_times_softmax(logits = logits, 
#                                                         mask = mask.squeeze(1).squeeze(1),
#                                                         topk = 25, #max(15, logits.shape[-1] // 2),
#                                                         temperature = 0.1)
#
#        # -> needs to be multiplied for gradient flow (just indexing does only take the gradient from the slice, but not the indexing var)
#        # now, logit_reducer's get grads after .backward()
#        #
#        # sampled_reps & mask now have sequence length of topk
#        if self.training: # not sure if keeping the empty slices is actually needed (could be for the gradient flow to those indices?)
#            sampled_reps = reps * gumbel_mask.unsqueeze(-1)
#            sampled_mask = mask
#            sampled_mask[gumbel_mask.bool().unsqueeze(1).unsqueeze(1) == 0] = -10000
#        else: # reduce the size for inference
#            sampled_reps = reps[gumbel_mask.bool(),:].view(reps.shape[0],-1,reps.shape[-1])
#            sampled_mask = mask[gumbel_mask.bool().unsqueeze(1).unsqueeze(1)].view(reps.shape[0],-1).unsqueeze(1).unsqueeze(1)
#
#        return sampled_reps, sampled_mask, gumbel_indices
#
#    # gumbel code from: https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
#    def sample_gumbel(self, shape,device, eps=1e-20):
#        U = torch.rand(shape,device=device)
#        return - torch.log(-torch.log(U + eps) + eps)
#
#    def gumbel_softmax_sample(self, logits, mask, temperature):
#
#        #rand baseline
#        #y = self.sample_gumbel(logits.size(),logits.device)
#
#        # todo: does this make sense? -> only add noise during training, keep inference deterministic
#        if self.training:
#            gumbel_noise = self.sample_gumbel(logits.size(), logits.device)
#            #gumbel_noise = gumbel_noise / (gumbel_noise.mean() / (logits.mean()+1e-3) + 1e-3)
#            y = logits + gumbel_noise
#        else:
#            y = logits
#
#        # add mask (-10000 entries for padding) to get a masked softmax
#        # -> discourage sampling from padding
#        y = y + mask
#        return torch.softmax(y / temperature, dim=-1)
#
#    def gumbel_softmax(self, logits, mask, topk, temperature):
#        """
#
#        applies the gumbel-softmax once and then takes the  and takes the argmax each time, masking previous picks
#
#        input: [*, n_class]
#        return: [*, n_class] boolean sample mask, indices selected
#        """
#        y = self.gumbel_softmax_sample(logits, mask, temperature)
#
#        #
#        # special first index cls vector handling, force to sample the first vector
#        # could be removed for tasks not depending on cls
#        # added here to not mess with the softmax
#        #
#        cls_force = torch.zeros_like(y)
#        cls_force[:,0] = 1
#        y = y + cls_force
#
#        shape = y.size()
#        _, ind = y.topk(topk, dim=-1)
#        y_hard = torch.zeros_like(y).view(-1, shape[-1])
#        y_hard.scatter_(1, ind, 1)
#        y_hard = y_hard.view(*shape)
#        return (y_hard - y).detach() + y , ind
#
#    def gumbel_k_times_softmax(self, logits, mask, topk, temperature):
#        """
#        
#        applies the gumbel-softmax k-times and takes the argmax each time, masking previous picks
#
#        input: [*, n_class]
#        return: [*, n_class] boolean sample mask, indices selected
#        """
#        #
#        # special first index cls vector handling, force to sample the first vector
#        # could be removed for tasks not depending on cls
#        #
#        logits[:,0] = 100 + logits.max(-1)[0]
#        top_k_mask = mask.clone()
#
#        y_hard = torch.zeros_like(logits)
#        y = torch.zeros_like(logits)
#        all_ind = []
#        for i in range(topk):
#            y += self.gumbel_softmax_sample(logits, top_k_mask, temperature)
#            y[y_hard == 1] = 0
#
#            ind = y.argmax(dim=-1)
#            all_ind.append(ind.unsqueeze(-1))
#            y_hard.scatter_(1, ind.unsqueeze(-1), 1)
#            y = (y_hard - y).detach() + y
#
#            top_k_mask[y_hard == 1] = -10000
#
#        return y, torch.cat(all_ind,dim=1)
#