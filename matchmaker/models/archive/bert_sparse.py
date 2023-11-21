#from transformers import *
#import math
#
#class Bert_Sparse(BertModel):
#
#    def __init__(self, config):
#        super().__init__(config)
#        self.encoder = BertSparseEncoder(config)
#
#    def reanimate(self,added_bias,layers):
#        self.encoder.reanimate(added_bias,layers)
#
#class BertSparseEncoder(BertEncoder):
#    def __init__(self, config):
#        super().__init__(config)
#        self.sparse_layer = nn.ModuleList([SparsityMaker(config) for _ in range(config.num_hidden_layers - 1)])
#
#    def reanimate(self,added_bias,layers):
#        for i,l in enumerate(self.sparse_layer):
#            if int(layers[i]) == 1:
#                l.reanimate(added_bias)
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
#            #if i == 5:
#            #    break
#
#            #
#            # Sparsity part
#            #
#            if i < len(self.sparse_layer) and i > 1:
#                hidden_states, sparsity_vecs = self.sparse_layer[i](hidden_states)
#                m_b = (attention_mask == 0).squeeze(1).squeeze(1)
#
#                all_sp_vecs.append(sparsity_vecs.squeeze(-1)[m_b] * (1/math.log2(2 + i))) # only minimize non-padding sparsity
#                all_sp_stats[i] = (((sparsity_vecs <= 0).float().squeeze(-1) * m_b).sum(dim=-1) / m_b.sum(-1)).mean()
#
#                empty = (sparsity_vecs == 0).squeeze(-1).unsqueeze(1).unsqueeze(1)
#                attention_mask[empty] = -10000
#
#                hidden_states,attention_mask = self.pack_tensor(hidden_states,attention_mask,sparsity_vecs)
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
#        return outputs + (all_sp_vecs,all_sp_stats,) # last-layer hidden state, (all hidden states), (all attentions)
#
#    def pack_tensor(self,tensor,mask,sparsity):
#        non_empty = (sparsity != 0).squeeze(-1) # todo what about padding?
#
#        all_lens = non_empty.sum(-1)
#        new_max_len = torch.max(all_lens)
#        new_indices = (torch.arange(new_max_len,device=tensor.device).unsqueeze(-1) < all_lens).t()
#
#        new_tensor = torch.zeros((tensor.shape[0],new_max_len,tensor.shape[-1]),device=tensor.device,dtype=tensor.dtype)
#        #new_tensor2 = torch.zeros((tensor.shape[0],new_max_len,tensor.shape[-1]),device=tensor.device,dtype=tensor.dtype)
#        new_mask = torch.zeros((tensor.shape[0],new_max_len),device=tensor.device,dtype=tensor.dtype)
#        new_mask += -10000
#        #for i in range(tensor.shape[0]):
#        #    new_tensor[i,:all_lens[i]] = tensor[i,non_empty[i],:]
#        #    new_mask[i,:,:,:all_lens[i]] = mask[i,:,:,non_empty[i]]
#
#        new_tensor[new_indices,:] = tensor[non_empty,:]
#        new_mask[new_indices] = mask[non_empty.unsqueeze(1).unsqueeze(1)]
#
#        return new_tensor,new_mask.unsqueeze(1).unsqueeze(1)
#
#
#class SparsityMaker(nn.Module):
#
#    def __init__(self, config):
#        super().__init__()
#        self.stop_word_reducer = nn.Linear(config.hidden_size, config.hidden_size // 2, bias=True)
#        self.stop_word_reducer2 = nn.Linear(config.hidden_size // 2, 1, bias=True)
#        torch.nn.init.constant_(self.stop_word_reducer2.bias, 1) # make sure we don't start in a broken state
#
#    def forward(self, reps):
#        sparsity_vec = torch.nn.functional.relu(torch.tanh(self.stop_word_reducer2(torch.tanh(self.stop_word_reducer(reps)))))
#        sparsity_vec[:,0] = 1
#        sparse_reps = reps * sparsity_vec
#        return sparse_reps, sparsity_vec
#
#    def reanimate(self,added_bias):
#        self.stop_word_reducer2.bias.data += added_bias
#        
#