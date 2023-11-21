from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          
from allennlp.modules.matrix_attention.dot_product_matrix_attention import *                          
import math
from matchmaker.modules.neuralIR_encoder import get_single_vectors_n_masks
import random

from matchmaker.modules.topic_attention_transformer import *
#from matchmaker.modules.multihead_dot_product import *

class TKL_MF(nn.Module):
    '''
    TKL is a neural IR model for long documents
    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim,padding_idx):
        return TKL_MF(word_embeddings_out_dim, 
                     kernels_mu =    config["tk_kernels_mu"],
                     kernels_sigma = config["tk_kernels_sigma"],
                     att_heads =     config["tk_att_heads"],
                     att_layer =     config["tk_att_layer"],
                     att_proj_dim =  config["tk_att_proj_dim"],
                     att_ff_dim =    config["tk_att_ff_dim"],
                     max_length =    config["max_doc_length"],
                     use_pos_encoding     = config["tk_use_pos_encoding"],
                     use_diff_posencoding = config["tk_use_diff_posencoding"],
                     saturation_type= config["tk_saturation_type"],
                     use_bert = config["token_embedder_type"] == "bert_vectors",
                     padding_idx=padding_idx
                     )

    def __init__(self,
                 _embsize:int,
                 kernels_mu: List[float],
                 kernels_sigma: List[float],
                 att_heads: int,
                 att_layer: int,
                 att_proj_dim: int,
                 att_ff_dim: int,
                 max_length,
                 use_pos_encoding,  
                 use_diff_posencoding,
                 saturation_type,
                use_bert,
                padding_idx
                ):

        super(TKL_MF, self).__init__()

        self.padding_idx = padding_idx
        n_kernels = len(kernels_mu)
        self.use_pos_encoding     = use_pos_encoding    
        self.use_diff_posencoding = use_diff_posencoding

        self.re_use_encoding = True

        self.chunk_size = 50
        self.overlap = 7
        self.extended_chunk_size = self.chunk_size + 2 * self.overlap
        
        self.sliding_window_size = 30
        self.top_k_chunks = 3
        self.use_bert = use_bert

        self.use_idf_sat = saturation_type == "idf"
        self.use_embedding_sat = saturation_type == "embedding"
        self.use_linear_sat = saturation_type == "linear"
        self.use_log_sat = saturation_type == "log"

        if len(kernels_mu) != len(kernels_sigma):
            raise Exception("len(kernels_mu) != len(kernels_sigma)")

        # static - kernel size & magnitude variables
        self.mu = nn.Parameter(torch.cuda.FloatTensor(kernels_mu), requires_grad=False)#.view(1, 1, 1, n_kernels)
        self.sigma = nn.Parameter(torch.cuda.FloatTensor(kernels_sigma), requires_grad=False)#.view(1, 1, 1, n_kernels)
        #self.mu.data.requires_grad=True
        #self.sigma.data.requires_grad=True

        pos_f = self.get_positional_features(_embsize, 32) #max_timescale=100000
        pos_f.requires_grad = True
        self.positional_features_q = nn.Parameter(pos_f)
        self.positional_features_q.requires_grad = True

        if self.use_diff_posencoding == True:
            pos_f = self.get_positional_features(_embsize,2000+500+self.extended_chunk_size)[:,500:,:].clone() #max_timescale=100000
            pos_f.requires_grad = True
            self.positional_features_d = nn.Parameter(pos_f)
            self.positional_features_d.requires_grad = True
        else:
            self.positional_features_d = self.positional_features_q

        pos_f = self.get_positional_features(_embsize, 100 + 70) #max_timescale=100000
        pos_f.requires_grad = True
        self.positional_features_t = nn.Parameter(pos_f[:,100:,:])
        self.positional_features_t.requires_grad = True


        self.title_encode = nn.Parameter(torch.empty((1,1,_embsize)),requires_grad=True)
        torch.nn.init.uniform_(self.title_encode, -0.014, 0.014)

        self.mixer = nn.Parameter(torch.full([1], 0.5, dtype=torch.float32, requires_grad=True))
        self.mixer_sat = nn.Parameter(torch.full([1], 0.5, dtype=torch.float32, requires_grad=True))

        #self.emb_reducer = nn.Linear(_embsize, 300, bias=True)

        self.title_score_extra = False
        self.topic_attention = True
        self.ta_type="cls-topic"

        if self.ta_type=="cls-topic":
            self.max_chunks = int(max_length / self.chunk_size + 1)
            self.cls_topic = nn.Parameter(torch.zeros([self.max_chunks,_embsize], dtype=torch.float32, requires_grad=True))
            torch.nn.init.xavier_normal_(self.cls_topic.data)
            #torch.nn.init.normal_(self.cls_topic.data)
            self.cls_topic.data /= torch.norm(self.cls_topic.data, p=2, dim=-1).unsqueeze(-1)

        if self.topic_attention:
            self.contextualizer = TransformerTA(d_model=_embsize, nhead=att_heads, num_encoder_layers=att_layer,
                 num_decoder_layers=att_layer, dim_feedforward=att_ff_dim, dropout=0,ta_type = self.ta_type,activation="gelu")
        else:
            encoder_layer = nn.TransformerEncoderLayer(_embsize, att_heads, dim_feedforward=att_ff_dim, dropout=0)
            self.contextualizer = nn.TransformerEncoder(encoder_layer, att_layer, norm=None)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        self.saturation_linear = nn.Linear(2, 1, bias=True)
        torch.nn.init.constant_(self.saturation_linear.bias, 100)
        torch.nn.init.uniform_(self.saturation_linear.weight, -0.014, 0.014)

        self.saturation_linear2 = nn.Linear(2, 1, bias=True)
        torch.nn.init.constant_(self.saturation_linear2.bias, 100)
        torch.nn.init.uniform_(self.saturation_linear2.weight, -0.014, 0.014)

        self.saturation_linear3 = nn.Linear(2, 1, bias=True)
        torch.nn.init.constant_(self.saturation_linear3.bias, 100)
        torch.nn.init.uniform_(self.saturation_linear3.weight, -0.014, 0.014)
        
        self.title_saturation_linear = nn.Linear(2, 1, bias=True)
        torch.nn.init.constant_(self.title_saturation_linear.bias, 100)
        torch.nn.init.uniform_(self.title_saturation_linear.weight, -0.014, 0.014)

        self.title_saturation_linear2 = nn.Linear(2, 1, bias=True)
        torch.nn.init.constant_(self.title_saturation_linear2.bias, 100)
        torch.nn.init.uniform_(self.title_saturation_linear2.weight, -0.014, 0.014)

        self.title_saturation_linear3 = nn.Linear(2, 1, bias=True)
        torch.nn.init.constant_(self.title_saturation_linear3.bias, 100)
        torch.nn.init.uniform_(self.title_saturation_linear3.weight, -0.014, 0.014)


        self.sat_normer = nn.LayerNorm(2,elementwise_affine=True)
        #self.sat_emb_reduce1 = nn.Linear(_embsize,_embsize, bias=False)
        self.sat_emb_reduce1 = nn.Linear(_embsize, 1, bias=False)
        #torch.nn.init.constant_(self.sat_emb_reduce1.bias, 2)

        self.kernel_mult_title = nn.Parameter(torch.full([1,1,n_kernels], 1, dtype=torch.float32, requires_grad=True))
        #self.length_normer = nn.Parameter(torch.full([1,1,1,1], 30, dtype=torch.float32, requires_grad=True))



        self.chunk_scoring = nn.Parameter(torch.full([1,self.top_k_chunks*5], 1, dtype=torch.float32, requires_grad=True))
        self.mixer_end = nn.Parameter(torch.full([1], 0.5, dtype=torch.float32, requires_grad=True))
        
        self.title_bin_weights = nn.Linear(n_kernels, 1, bias=False)
        torch.nn.init.uniform_(self.title_bin_weights.weight, -0.014, 0.014) # inits taken from matchzoo
        
        self.title_mix = nn.Linear(2, 1, bias=False)
        torch.nn.init.constant_(self.title_mix.weight, 1)

        self.dense = nn.Linear(n_kernels, 1, bias=False)
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014) # inits taken from matchzoo
        
        self.multihead_dot = MultiheadDotProduct(_embsize, att_heads)

        self.part_mix =  nn.Parameter(torch.full([1,8], 1, dtype=torch.float32, requires_grad=True))

        self.head_mix = nn.Linear(8, 1, bias=False)
        self.cls_pooler = nn.Linear(_embsize, _embsize)
        torch.nn.init.uniform_(self.cls_pooler.weight, -0.001, 0.001)
        #self.cls_normer = nn.LayerNorm(200)

    def forward(self, word_embeddings, query, document, title, output_secondary_output) -> torch.Tensor:
        # pylint: disable=arguments-differ
        query_embeddings,query_mask = self.forward_representation(word_embeddings, query,None, "query")

        documents_cls,parts_cls,documents_unique_again,document_mask_packed_unique,packed_indices,chunk_pieces,total_chunks = self.forward_representation(word_embeddings, document, title, "doc_model")

        query_vecs = self.cls_pooler(query_embeddings[:,0,:])
        query_vecs = query_vecs / (query_vecs.norm(p=2,dim=1, keepdim=True) + 0.0001)
        
        document_main_cls = self.cls_pooler(documents_cls[:,0,:])
        document_main_cls = document_main_cls / (document_main_cls.norm(p=2,dim=1, keepdim=True) + 0.0001)
#
        #score = query_vecs.bmm(documents_cls)
        #score = nn.CosineSimilarity(dim=1, eps=1e-6)(query_vecs,document_main_cls)
        #scaling = float(query_vecs.shape[-1]) ** -0.5
        #score = torch.bmm(query_vecs.unsqueeze(dim=1)*scaling, document_main_cls.unsqueeze(dim=2)).squeeze(-1).squeeze(-1)
        score = torch.bmm(query_vecs.unsqueeze(dim=1), document_main_cls.unsqueeze(dim=2)).squeeze(-1).squeeze(-1) * 6

        #score = torch.exp((query_vecs * document_main_cls).sum(-1) + 1e-6)
        #score = torch.bmm(query_vecs.unsqueeze(dim=1), parts_cls.transpose(2,1)).squeeze(1)
        #score=score / 300
        #score[score == 0] = -1000
#
        #score = score.sort(dim=1,descending=True)[0] * self.part_mix[:,:score.shape[1]]
#
        #score[score <= -900] = 0

        #score = self.multihead_dot(query_vecs.unsqueeze(dim=0),document_main_cls.unsqueeze(dim=0))
        
        #score = score.squeeze(-1).squeeze(-1)

        #score = score.sort(dim=1,descending=True)[0] * self.part_mix[:,:score.shape[1]]
        #score = self.head_mix(score).squeeze(-1)
        #score = score.sum(-1)

                


        if output_secondary_output:
            return (score, {}), query_vecs, document_main_cls
        else:
            return score,query_vecs,document_main_cls

        #
        # cosine matrix
        # -------------------------------------------------------
        packed_query_embeddings = query_embeddings.unsqueeze(1).expand(-1,chunk_pieces,-1,-1).reshape(-1,query_embeddings.shape[1],query_embeddings.shape[-1])[packed_indices]
        #packed_query_mask = query_pad_oov_mask.unsqueeze(1).expand(-1,chunk_pieces,-1).reshape(-1,query_embeddings.shape[1])[packed_indices]
        #if packed_query_embeddings.shape[0] % 8 != 0:
        #    packed_query_embeddings = nn.functional.pad(packed_query_embeddings,(0,0,0,0,0, 8 - packed_query_embeddings.shape[0] % 8),value=self.padding_idx)
 
        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(packed_query_embeddings, documents_unique_again)
 
        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
 
        cosine_matrix_extradim = cosine_matrix.unsqueeze(-1)        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu.view(1, 1, 1, -1), 2) / (2 * torch.pow(self.sigma.view(1, 1, 1, -1), 2)))
        kernel_results_masked = raw_kernel_results * document_mask_packed_unique.unsqueeze(1).unsqueeze(-1)
 
        kerne_activations_per_doc = torch.zeros((total_chunks,query_embeddings.shape[1],documents_unique_again.shape[1],kernel_results_masked.shape[-1]), dtype=documents_unique_again.dtype, layout=documents_unique_again.layout, device=documents_unique_again.device)
        kerne_activations_per_doc[packed_indices] = kernel_results_masked#[:ids_packed_orig_bs]
 
        kerne_activations_per_doc = kerne_activations_per_doc.transpose(1,2).reshape(query_embeddings.shape[0],-1,query_embeddings.shape[1],kernel_results_masked.shape[-1]).transpose(2,1)
 
 
        #
        # kernel-pooling
        # -------------------------------------------------------
 
        if kerne_activations_per_doc.shape[2] < self.sliding_window_size:
            kerne_activations_per_doc = nn.functional.pad(kerne_activations_per_doc,(0,0,0, self.sliding_window_size - kerne_activations_per_doc.shape[2]))
 
        unrolled_kernel_activations = kerne_activations_per_doc.unfold(2,self.sliding_window_size,2).transpose(-1,-2)
        unrolled_kernel_activation_lengths = torch.sum(unrolled_kernel_activations.sum(dim=-1) != 0,dim=-1)
        per_kernel_query = torch.sum(unrolled_kernel_activations, -2) 
 
        sat_influencer = torch.cat([torch.relu(self.sat_emb_reduce1(query_embeddings)).expand_as(unrolled_kernel_activation_lengths).unsqueeze(-1),
                                    unrolled_kernel_activation_lengths.float().unsqueeze(-1)],dim=-1)
 
        sat_influencer = self.sat_normer(sat_influencer)
 
        sat1 = self.saturation_linear(sat_influencer)
        sat2 = 1 / self.saturation_linear2(sat_influencer)
        sat3 = self.saturation_linear3(sat_influencer)
 
        sat_per_kernel_query = sat1 * (torch.clamp(per_kernel_query, min=1e-10) ** sat2) - sat3
 
        sat_per_kernel_query = sat_per_kernel_query * query_mask.unsqueeze(-1).unsqueeze(-1) * (unrolled_kernel_activation_lengths > 0).float().unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(sat_per_kernel_query, 1) 
 
        dense_out = self.dense(per_kernel)
        score = dense_out.squeeze(-1)
 
        if score.shape[1] < self.top_k_chunks:
            score = nn.functional.pad(score,(0, self.top_k_chunks - score.shape[1]))
 
        score[score == 0] = -9900
        orig_score = score
 
        #
        # argmax top-n hills
        # 
        top_non_overlapping_idx = torch.zeros((orig_score.shape[0],self.top_k_chunks), dtype=torch.long, device=orig_score.device) 
        max_per_region_score = orig_score.clone()
 
        r = torch.arange(max_per_region_score.shape[1],device=max_per_region_score.device)
 
        for c in range(0,self.top_k_chunks):
           
            best_index = torch.argmax(max_per_region_score,dim=1)
            top_non_overlapping_idx[:,c] = best_index
            region_pool = torch.abs(r - best_index.unsqueeze(-1)) < self.sliding_window_size / 2
            max_per_region_score[region_pool] = -10001 - c
 
        top_non_overlapping_idx_neighbors = torch.cat([top_non_overlapping_idx,top_non_overlapping_idx - 1,top_non_overlapping_idx + 1,top_non_overlapping_idx - 2,top_non_overlapping_idx + 2],dim=1)
        top_non_overlapping_idx_neighbors[top_non_overlapping_idx_neighbors < 0] = 0
        top_non_overlapping_idx_neighbors[top_non_overlapping_idx_neighbors >= orig_score.shape[1]] = orig_score.shape[1] - 1
 
        topk_indices_flat = (top_non_overlapping_idx_neighbors + torch.arange(0,orig_score.shape[0]*orig_score.shape[1],orig_score.shape[1],device=orig_score.device).unsqueeze(-1)).view(-1)
        top_k_non_overlapping = orig_score.view(-1).index_select(0,topk_indices_flat).view(top_non_overlapping_idx.shape[0],-1)
        top_k_non_overlapping[top_k_non_overlapping <= -9900] = 0
 
        orig_score[orig_score <= -9900] = 0
 
        score = (top_k_non_overlapping * self.chunk_scoring).sum(dim=1)
 
        # title score 
        if self.title_score_extra == True:
            #title_ids = title["tokens"]["tokens"]
            #title_embs = word_embeddings.token_embedder_tokens(title_ids)
            #title_embs = title_embs + self.positional_features_q[:,:title_embs.shape[1]]
            #title_embs = self.contextualizer((title_embs).transpose(1,0),src_key_padding_mask=~title["tokens"]["mask"].bool()).transpose(1,0)
            title_embs = documents_cls

            cosine_matrix = self.cosine_module.forward(query_embeddings, title_embs).unsqueeze(-1)
            raw_kernel_results = torch.exp(- torch.pow(cosine_matrix - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
 
            per_kernel_query = torch.sum(raw_kernel_results, 2)
 
            #a = torch.relu(self.sat_emb_reduce1(query_embeddings))
            #sat_influencer = torch.cat([a, torch.full_like(a,title_embs.shape[1])],dim=-1)
 #
            #sat1 = self.title_saturation_linear(sat_influencer)
            #sat2 = 1 / self.title_saturation_linear2(sat_influencer)
            #sat3 = self.title_saturation_linear3(sat_influencer)
 #
            #sat_per_kernel_query = sat1 * (torch.clamp(per_kernel_query, min=1e-10) ** sat2) - sat3
            log_per_kernel_query = torch.log(torch.clamp(per_kernel_query * self.kernel_mult_title, min=1e-10))
            #log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) #* 0.01
            log_per_kernel_query_masked = log_per_kernel_query * query_mask.unsqueeze(-1) # make sure we mask out padding values
 
            per_kernel = torch.sum(log_per_kernel_query_masked, 1) 
            title_score = self.title_bin_weights(per_kernel)
 
            score = self.title_mix(torch.cat([title_score,score.unsqueeze(-1)],dim=1)).squeeze(-1)
 
        if output_secondary_output:
            query_mean_vector = query_embeddings.sum(dim=1) / query_pad_oov_mask.sum(dim=1).unsqueeze(-1)
            sat_influence_from_top_k = sat_influencer.transpose(1,2).reshape(-1,query_embeddings.shape[1],2).index_select(0,topk_indices_flat).view(top_non_overlapping_idx_neighbors.shape[0],top_non_overlapping_idx_neighbors.shape[1],query_embeddings.shape[1],2)
            return score, {"score":score,"orig_score":orig_score,"top_non_overlapping_idx":top_non_overlapping_idx,"orig_doc_len":(document_ids > 0).sum(dim=-1),"top_k_non_overlapping":top_k_non_overlapping,"sat_influence_from_top_k":sat_influence_from_top_k,
                           "total_chunks":chunked_ids_unrolled.shape[0],"packed_chunks":documents_packed.shape[0]}
        else:
            return score

    def forward_representation(self, word_embeddings, sequence: torch.Tensor, title: torch.Tensor, sequence_type:str) -> torch.Tensor:

        if sequence_type == "query":
            query_embeddings, query_mask = get_single_vectors_n_masks(word_embeddings, sequence)
            
            pos_sequence = query_embeddings + self.positional_features_t[:,:query_embeddings.shape[1]]
            return self.contextualizer.forward_query((pos_sequence).transpose(1,0),src_key_padding_mask=~query_mask.bool()).transpose(1,0),query_mask

        elif sequence_type == "pretrain" or sequence_type == "doc_model":

            document_ids = sequence["tokens"]["tokens"]
            orig_shape = document_ids.shape

            if document_ids.shape[1] > self.overlap:
                #needed_padding = self.extended_chunk_size - ((document_ids.shape[1] - self.overlap) % self.chunk_size)
                needed_padding = self.extended_chunk_size - (((document_ids.shape[1]) % self.chunk_size)  - self.overlap)
            else:
                needed_padding = self.extended_chunk_size - self.overlap - document_ids.shape[1]
            orig_doc_len = document_ids.shape[1]

            document_ids = nn.functional.pad(document_ids,(self.overlap, needed_padding),value=self.padding_idx)

            chunked_ids = document_ids.unfold(1,self.extended_chunk_size,self.chunk_size)

            batch_size = chunked_ids.shape[0]
            chunk_pieces = chunked_ids.shape[1]

            chunked_ids_unrolled=chunked_ids.reshape(-1,self.extended_chunk_size)
            #packed_indices = (chunked_ids_unrolled[:,self.overlap:-self.overlap] == self.padding_idx).sum(-1) == 0 #+title_size
            packed_indices = (chunked_ids_unrolled[:,self.overlap:-self.overlap] != self.padding_idx).any(-1)
            ids_packed = chunked_ids_unrolled[packed_indices]
            ids_packed_orig_bs = ids_packed.shape[0]

            #if ids_packed.shape[0] % 8 != 0:
            #    ids_packed = nn.functional.pad(ids_packed,(0,0,0, 8 - ids_packed.shape[0] % 8),value=self.padding_idx)

            if not self.use_bert:
                documents_packed = word_embeddings.token_embedder_tokens(ids_packed)
            else:
                documents_packed = word_embeddings.token_embedder_tokens(ids_packed,token_type_ids=torch.ones_like(ids_packed))

            padding_packed = (ids_packed != self.padding_idx).float()


            document_pos_encoding = self.positional_features_d[:,:documents_packed.shape[1],:]
            pos_sequence = documents_packed + document_pos_encoding

            if self.topic_attention:
                title_ids = title["tokens"]["tokens"]
                title_embs = word_embeddings.token_embedder_tokens(title_ids)
                title_embs = title_embs + self.positional_features_t[:,:title_embs.shape[1]]
                #title_non_zero_indices = title["tokens"]["mask"].sum(-1) > 0
                #title_non_zero = title_embs[title_non_zero_indices]
                #title_non_zero_mask = title["tokens"]["mask"][title_non_zero_indices]
                #title_embs[title_non_zero_indices] = self.contextualizer((title_non_zero).transpose(1,0),src_key_padding_mask=~title_non_zero_mask.bool()).transpose(1,0)

                if self.ta_type == "maxpool-topic" or self.ta_type == "cls-topic":
                    packed_title_embeddings = title_embs
                    packed_title_mask = title["tokens"]["mask"]
                else: # title -only, so expand right now
                    packed_title_embeddings = title_embs.unsqueeze(1).expand(-1,chunk_pieces,-1,-1).reshape(-1,title_embs.shape[1],title_embs.shape[-1])[packed_indices]
                    packed_title_mask = title["tokens"]["mask"].unsqueeze(1).expand(-1,chunk_pieces,-1).reshape(-1,title_embs.shape[1])[packed_indices]
                #pos_sequence = torch.cat([packed_title_embeddings, pos_sequence],1)
                #padding_packed = torch.cat([packed_title_mask, padding_packed],1)
                ts = (chunked_ids_unrolled.shape[0],1,documents_packed.shape[-1])

                if self.ta_type=="cls-topic":
                    packed_cls_embeddings = self.cls_topic[:chunk_pieces].unsqueeze(1).unsqueeze(0).expand(batch_size,-1,-1,-1).reshape(-1,1,self.cls_topic.shape[-1])[packed_indices]
                    #packed_cls_embeddings = self.cls_topic.unsqueeze(0).unsqueeze(0).expand(pos_sequence.shape[0],-1,-1)
                    pos_sequence = torch.cat([packed_cls_embeddings, pos_sequence],1)
                    padding_packed = torch.cat([torch.ones([padding_packed.shape[0],1],device=padding_packed.device), padding_packed],1)

                documents_packed, topic_embs, title_embs  = self.contextualizer(tgt=(pos_sequence).transpose(1,0),src=(packed_title_embeddings).transpose(1,0),
                                                       tgt_key_padding_mask=~padding_packed.bool(),src_key_padding_mask=~packed_title_mask.bool(),memory_key_padding_mask=~packed_title_mask.bool(),
                                                       tgt_global_shape=ts,tgt_packed_indices=packed_indices,chunk_pieces=chunk_pieces,mode=sequence_type)
                documents_packed = documents_packed.transpose(1,0)
            else:
                documents_packed = self.contextualizer((pos_sequence).transpose(1,0),src_key_padding_mask=~padding_packed.bool()).transpose(1,0)
                topic_embs=None
                title_embs=None
                parts_cls=None

            if sequence_type == "pretrain":
                if self.ta_type=="cls-topic":
                    documents_unique_again = documents_packed[:,self.overlap+1:-self.overlap,:]
                    documents_parts_cls = documents_packed[:,0,:] # the cls vec
                
                    unpacked_parts_cls = torch.zeros((chunked_ids_unrolled.shape[0],documents_packed.shape[-1]), 
                                                 dtype=documents_packed.dtype, layout=chunked_ids_unrolled.layout, device=chunked_ids_unrolled.device)
                    unpacked_parts_cls[packed_indices] = documents_parts_cls
                    parts_cls = unpacked_parts_cls.view(batch_size,-1,documents_packed.shape[-1])

                else:
                    documents_unique_again = documents_packed[:,self.overlap:-self.overlap,:]
                    parts_cls=None


                unpacked_documents = torch.zeros((chunked_ids_unrolled.shape[0],documents_unique_again.shape[1],documents_packed.shape[-1]), 
                                                 dtype=documents_packed.dtype, layout=chunked_ids_unrolled.layout, device=chunked_ids_unrolled.device)
                unpacked_documents[packed_indices] = documents_unique_again
                document_embeddings = unpacked_documents.view(batch_size,-1,self.chunk_size,documents_packed.shape[-1])\
                                                        .view(batch_size,-1,documents_packed.shape[-1])[:,:orig_shape[1],:]

                return document_embeddings,topic_embs,title_embs,parts_cls,packed_indices.view(-1,chunk_pieces)
            elif sequence_type == "doc_model":
                if self.ta_type=="cls-topic":
                    documents_parts_cls = documents_packed[:,0,:] # the cls vec
                
                    unpacked_parts_cls = torch.zeros((chunked_ids_unrolled.shape[0],documents_packed.shape[-1]), 
                                                 dtype=documents_packed.dtype, layout=chunked_ids_unrolled.layout, device=chunked_ids_unrolled.device)
                    unpacked_parts_cls[packed_indices] = documents_parts_cls
                    parts_cls = unpacked_parts_cls.view(batch_size,-1,documents_packed.shape[-1])

                    return topic_embs, parts_cls, documents_packed[:,self.overlap:-self.overlap,:], padding_packed[:,self.overlap:-self.overlap],packed_indices,chunk_pieces,chunked_ids_unrolled.shape[0]
                else:
                    return documents_packed[:,self.overlap:-self.overlap,:], padding_packed[:,self.overlap:-self.overlap],packed_indices,chunk_pieces,chunked_ids_unrolled.shape[0]

    def get_positional_features(self,dimensions,
                                max_length,
                                min_timescale: float = 1.0,
                                max_timescale: float = 1.0e4):
        # pylint: disable=line-too-long
        """
        Implements the frequency-based positional encoding described
        in `Attention is all you Need
        <https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .

        Adds sinusoids of different frequencies to a ``Tensor``. A sinusoid of a
        different frequency and phase is added to each dimension of the input ``Tensor``.
        This allows the attention heads to use absolute and relative positions.

        The number of timescales is equal to hidden_dim / 2 within the range
        (min_timescale, max_timescale). For each timescale, the two sinusoidal
        signals sin(timestep / timescale) and cos(timestep / timescale) are
        generated and concatenated along the hidden_dim dimension.

        Parameters
        ----------
        tensor : ``torch.Tensor``
            a Tensor with shape (batch_size, timesteps, hidden_dim).
        min_timescale : ``float``, optional (default = 1.0)
            The smallest timescale to use.
        max_timescale : ``float``, optional (default = 1.0e4)
            The largest timescale to use.

        Returns
        -------
        The input tensor augmented with the sinusoidal frequencies.
        """
        timesteps=max_length
        hidden_dim = dimensions

        timestep_range = self.get_range_vector(timesteps, 0).data.float()
        # We're generating both cos and sin frequencies,
        # so half for each.
        num_timescales = hidden_dim // 2
        timescale_range = self.get_range_vector(num_timescales, 0).data.float()

        log_timescale_increments = math.log(float(max_timescale) / float(min_timescale)) / float(num_timescales - 1)
        inverse_timescales = min_timescale * torch.exp(timescale_range * -log_timescale_increments)

        # Broadcasted multiplication - shape (timesteps, num_timescales)
        scaled_time = timestep_range.unsqueeze(1) * inverse_timescales.unsqueeze(0)
        # shape (timesteps, 2 * num_timescales)
        sinusoids = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 1)
        if hidden_dim % 2 != 0:
            # if the number of dimensions is odd, the cos and sin
            # timescales had size (hidden_dim - 1) / 2, so we need
            # to add a row of zeros to make up the difference.
            sinusoids = torch.cat([sinusoids, sinusoids.new_zeros(timesteps, 1)], 1)
        return sinusoids.unsqueeze(0)

    def get_range_vector(self, size: int, device: int) -> torch.Tensor:
        """
        Returns a range vector with the desired size, starting at 0. The CUDA implementation
        is meant to avoid copy data from CPU to GPU.
        """
        if device > -1:
            return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
        else:
            return torch.arange(0, size, dtype=torch.long)

    def get_param_stats(self): #" b: "+str(self.dense.bias.data) +\ "b: "+str(self.dense_mean.bias.data) +#"scaler: "+str(self.nn_scaler.data) +\ # " bias: " +str(self.saturation_linear.bias.data) +\
        return "TK: part_mix: "+str(self.part_mix.data) +\
        " self.title_bin_weights: " +str(self.title_bin_weights.weight.data) +\
        " self.title_mix: " +str(self.title_mix.weight.data) +\
        " self.chunk_scoring: " +str(self.chunk_scoring.data) +\
        " self.kernel_mult_title: " +str(self.kernel_mult_title.data) +\
        " self.saturation_linear: " +str(self.saturation_linear.weight.data) + " bias: " +str(self.saturation_linear.bias.data) +\
        " self.saturation_linear2: " +str(self.saturation_linear2.weight.data) + " bias: " +str(self.saturation_linear2.bias.data) +\
        " self.saturation_linear3: " +str(self.saturation_linear3.weight.data) + " bias: " +str(self.saturation_linear3.bias.data) +\
        " self.title_saturation_linear: " +str(self.title_saturation_linear.weight.data) + " bias: " +str(self.title_saturation_linear.bias.data) +\
        " self.title_saturation_linear2: " +str(self.title_saturation_linear2.weight.data) + " bias: " +str(self.title_saturation_linear2.bias.data) +\
        " self.title_saturation_linear3: " +str(self.title_saturation_linear3.weight.data) + " bias: " +str(self.title_saturation_linear3.bias.data) +\
        "mixer: "+str(self.mixer.data) #+ "mixer_end: "+str(self.mixer_end.data)

    def get_param_secondary(self):
        return {"dense_weight":self.dense.weight,
                "saturation_linear_weight":self.saturation_linear.weight,
                "saturation_linear_bias":self.saturation_linear.bias,
                "saturation_linear2_weight":self.saturation_linear2.weight,
                "saturation_linear2_bias":self.saturation_linear2.bias,
                "saturation_linear3_weight":self.saturation_linear3.weight,
                "saturation_linear3_bias":self.saturation_linear3.bias,
                "chunk_scoring":self.chunk_scoring,
                "kernel_mult":self.kernel_mult,
                "mixer":self.mixer}