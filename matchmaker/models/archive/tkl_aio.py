from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          
from allennlp.modules.matrix_attention.dot_product_matrix_attention import *                          
import math

from matchmaker.modules.neuralIR_encoder import get_single_vectors_n_masks

class TKL_aio_1(nn.Module):
    '''
    TK is a neural IR model - a fusion between transformer contextualization & kernel-based scoring

    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return TKL_aio_1(word_embeddings_out_dim, 
                     kernels_mu =    config["tk_kernels_mu"],
                     kernels_sigma = config["tk_kernels_sigma"],
                     att_heads =     config["tk_att_heads"],
                     att_layer =     config["tk_att_layer"],
                     att_proj_dim =  config["tk_att_proj_dim"],
                     att_ff_dim =    config["tk_att_ff_dim"],
                     max_length =    config["max_doc_length"],
                     use_pos_encoding     = config["tk_use_pos_encoding"],
                     use_diff_posencoding = config["tk_use_diff_posencoding"],
                     use_bert = config["token_embedder_type"] == "bert_vectors"
                     #position_bias_bin_percent   = config["tk_position_bias_bin_percent"],
                     #position_bias_absolute_steps= config["tk_position_bias_absolute_steps"] 
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
                 #use_position_bias,
                 use_diff_posencoding,
                 use_bert
                 #position_bias_bin_percent,
                 #position_bias_absolute_steps
                 ):

        super(TKL_aio_1, self).__init__()

        n_kernels = len(kernels_mu)
        self.use_pos_encoding     = use_pos_encoding    
        self.use_diff_posencoding = use_diff_posencoding

        self.re_use_encoding = True
        self.use_bert = use_bert
        self.chunk_size = 40
        self.overlap = 5
        self.extended_chunk_size = self.chunk_size + 2 * self.overlap
        
        self.sliding_window_size = 30
        self.top_k_chunks = 3

        if len(kernels_mu) != len(kernels_sigma):
            raise Exception("len(kernels_mu) != len(kernels_sigma)")

        # static - kernel size & magnitude variables
        self.mu = nn.Parameter(torch.cuda.FloatTensor(kernels_mu), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = nn.Parameter(torch.cuda.FloatTensor(kernels_sigma), requires_grad=False).view(1, 1, 1, n_kernels)


        #
        # embedding & contextualization
        #

        pos_f = self.get_positional_features(_embsize, 30) #max_timescale=100000
        pos_f.requires_grad = True
        self.positional_features_q = nn.Parameter(pos_f)
        self.positional_features_q.requires_grad = True

        if self.use_diff_posencoding == True:
            pos_f = self.get_positional_features(_embsize,max_length+500+self.extended_chunk_size)[:,500:,:].clone() #max_timescale=100000
            pos_f.requires_grad = True
            self.positional_features_d = nn.Parameter(pos_f)
            self.positional_features_d.requires_grad = True
        else:
            self.positional_features_d = self.positional_features_q


        self.mixer = nn.Parameter(torch.full([1], 0.5, dtype=torch.float32, requires_grad=True))

        #self.emb_reducer = nn.Linear(_embsize, 100, bias=True)
        #self.emb_reducer_doc = copy.deepcopy(self.emb_reducer)
        #self.emb_drop = nn.Dropout(0.1)

        if not use_bert:
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
        
        self.sat_normer = nn.LayerNorm(2,elementwise_affine=True)

        #self.kernel_mult = nn.Parameter(torch.full([4,1,1,1,n_kernels], 1, dtype=torch.float32, requires_grad=True))
        #self.length_normer = nn.Parameter(torch.full([1,1,1,1], 30, dtype=torch.float32, requires_grad=True))

        self.chunk_scoring = nn.Parameter(torch.full([1,self.top_k_chunks*5], 1, dtype=torch.float32, requires_grad=True))

        self.dense = nn.Linear(n_kernels, 1, bias=False)
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014) # inits taken from matchzoo

    def forward(self, word_embeddings, word_embeddings_idfs, query, document, output_secondary_output) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # contextualization
        # -------------------------------------------------------

        query_embeddings, query_pad_oov_mask = get_single_vectors_n_masks(word_embeddings, query)
        if not self.use_bert:
            query_embeddings = self.forward_representation(query_embeddings, query_pad_oov_mask, self.positional_features_q[:,:query_embeddings.shape[1],:])


        document_ids = document["tokens"]

        if document_ids.shape[1] > self.overlap:
            needed_padding = self.extended_chunk_size - ((document_ids.shape[1] - self.overlap) % self.chunk_size)
        else:
            needed_padding = self.extended_chunk_size - self.overlap - document_ids.shape[1]
        orig_doc_len = document_ids.shape[1]


        #document_embeddings = nn.functional.pad(document_embeddings,(0,0,self.overlap, needed_padding))
        document_ids = nn.functional.pad(document_ids,(self.overlap, needed_padding))

        #chunked_docs = document_embeddings.unfold(1,self.extended_chunk_size,self.chunk_size).transpose(-1,-2)#[:,:,overlap:-overlap,:]
        chunked_ids = document_ids.unfold(1,self.extended_chunk_size,self.chunk_size)
        
        batch_size = chunked_ids.shape[0]
        chunk_pieces = chunked_ids.shape[1]

        #chunked_docs2=chunked_docs.reshape(-1,self.extended_chunk_size,document_embeddings.shape[-1])
        chunked_ids_unrolled=chunked_ids.reshape(-1,self.extended_chunk_size)
        #chunked_pad2[chunked_pad2.sum(-1) == 0] = 1

        packed_indices = chunked_ids_unrolled[:,self.overlap:-self.overlap].sum(-1) != 0
        ids_packed = chunked_ids_unrolled[packed_indices]

        if not self.use_bert:
            documents_packed = word_embeddings.token_embedder_tokens(ids_packed)
        else:
            documents_packed = word_embeddings.token_embedder_tokens(ids_packed,token_type_ids=torch.ones_like(ids_packed))

        padding_packed = (ids_packed > 0).float()
        #padding_packed = chunked_pad2[packed_indices]

        #if self.use_pos_encoding:
        #    documents_packed = documents_packed + self.positional_features_d[:,:documents_packed.shape[1],:]

        #documents_packed = self.emb_reducer_doc(documents_packed)
        if not self.use_bert:
            if self.use_pos_encoding:
                if self.re_use_encoding:
                    document_pos_encoding = self.positional_features_d[:,:documents_packed.shape[1],:]
                else:
                    document_pos_encoding = self.positional_features_d[:,:document_embeddings.shape[1],:]
                    document_pos_encoding = document_pos_encoding.unfold(1,self.extended_chunk_size,self.chunk_size).transpose(-1,-2)
                    document_pos_encoding = document_pos_encoding.squeeze(0)
                    document_pos_encoding = document_pos_encoding.repeat(document_embeddings.shape[0],1,1)[packed_indices]
            else:
                document_pos_encoding = None


            documents_packed = self.forward_representation(documents_packed, padding_packed, document_pos_encoding)

        documents_unique_again = documents_packed[:,self.overlap:-self.overlap,:]
        #document_mask_unique_again = chunked_pad[:,:,overlap:-overlap]
        document_mask_packed_unique = padding_packed[:,self.overlap:-self.overlap]
        #        
        # reshape back in original form
 
        #unpacked_documents = torch.zeros((chunked_docs2.shape[0],documents_unique_again.shape[1],chunked_docs2.shape[2]), dtype=chunked_docs2.dtype, layout=chunked_docs2.layout, device=chunked_docs2.device)
        #unpacked_documents[packed_indices] = documents_unique_again

        #document_embeddings = unpacked_documents.view(batch_size,-1,chunk_size,document_embeddings.shape[-1]).view(batch_size,-1,document_embeddings.shape[-1])
        #updated_mask = chunked_pad[:,:,overlap:-overlap].view(batch_size,-1)
        #assert not torch.isnan(documents_packed).any()

        #
        # masks 
        # -------------------------------------------------------

        #query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        #query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------
        packed_query_embeddings = query_embeddings.unsqueeze(1).expand(-1,chunk_pieces,-1,-1).reshape(-1,query_embeddings.shape[1],query_embeddings.shape[-1])[packed_indices]
        packed_query_mask = query_pad_oov_mask.unsqueeze(1).expand(-1,chunk_pieces,-1).reshape(-1,query_embeddings.shape[1])[packed_indices]

        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(packed_query_embeddings, documents_unique_again)
        #cosine_matrix_masked = cosine_matrix * query_by_doc_mask

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------

        cosine_matrix_extradim = cosine_matrix.unsqueeze(-1)        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * document_mask_packed_unique.unsqueeze(1).unsqueeze(-1)


        kerne_activations_per_doc = torch.zeros((chunked_ids_unrolled.shape[0],query_embeddings.shape[1],documents_unique_again.shape[1],kernel_results_masked.shape[-1]), dtype=torch.float, device=chunked_ids_unrolled.device)
        kerne_activations_per_doc[packed_indices] = kernel_results_masked

        #kerne_activations_per_doc = kerne_activations_per_doc.view(batch_size,query_embeddings.shape[1],-1,kernel_results_masked.shape[-1]).view(batch_size,-1,document_embeddings.shape[-1])
        kerne_activations_per_doc = kerne_activations_per_doc.transpose(1,2).reshape(batch_size,-1,query_embeddings.shape[1],kernel_results_masked.shape[-1]).transpose(2,1)


        #
        # kernel-pooling
        # -------------------------------------------------------
        unrolled_kernel_activations = kerne_activations_per_doc.unfold(2,self.sliding_window_size,2).transpose(-1,-2)
        unrolled_kernel_activation_lengths = torch.sum(unrolled_kernel_activations.sum(dim=-1) != 0,dim=-1)
        per_kernel_query = torch.sum(unrolled_kernel_activations, -2) 

        #query_idfs = word_embeddings_idfs(query)
        #sat_influencer = torch.cat([torch.tanh(query_idfs.expand_as(unrolled_kernel_activation_lengths).unsqueeze(-1)),
        #                           unrolled_kernel_activation_lengths.float().unsqueeze(-1)],dim=-1)
        #sat_influencer = torch.cat([query_embeddings.norm(p=2, dim=-1, keepdim=True).expand_as(unrolled_kernel_activation_lengths).unsqueeze(-1),
        #                            unrolled_kernel_activation_lengths.float().unsqueeze(-1)],dim=-1)
#
        #sat_influencer = self.sat_normer(sat_influencer)

        #sat1 = self.saturation_linear(sat_influencer)
        #sat2 = 1 / self.saturation_linear2(sat_influencer)
        #sat3 = self.saturation_linear3(sat_influencer)

        #log_per_kernel_query = sat1 * (torch.clamp(per_kernel_query, min=1e-10) ** sat2) - sat3
        #log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10))


        #sat1 = self.saturation_linear(sat_influencer)
        #sat2 = 1 / (self.saturation_linear2(sat_influencer)) #* self.kernel_mult[2])
        #sat3 = self.saturation_linear3(sat_influencer)

        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10))

        #log_per_kernel_query = sat1 * (torch.clamp(per_kernel_query, min=1e-10) ** sat2) - sat3
        #log_per_kernel_query = sat1 * self.kernel_mult[0] * (torch.clamp(per_kernel_query * self.kernel_mult[1], min=1e-10) ** sat2) - (sat3 * self.kernel_mult[3])

        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1).unsqueeze(-1) * (unrolled_kernel_activation_lengths > 0).float().unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        dense_out = self.dense(per_kernel)
        score = dense_out.squeeze(-1)

        if score.shape[1] < self.top_k_chunks:
            score = nn.functional.pad(score,(0, self.top_k_chunks - score.shape[1]))

        score[score == 0] = -9900
        orig_score = score

        #
        # argmax alternative
        # 
        top_non_overlapping_idx = torch.zeros((orig_score.shape[0],self.top_k_chunks), dtype=torch.long, device=orig_score.device) 
        max_per_region_score = orig_score.clone()

        r = torch.arange(max_per_region_score.shape[1],device=max_per_region_score.device)

        for c in range(0,self.top_k_chunks):
           
            best_index = torch.argmax(max_per_region_score,dim=1)
            top_non_overlapping_idx[:,c] = best_index
            region_pool = torch.abs(r - best_index.unsqueeze(-1)) < self.sliding_window_size / 2
            max_per_region_score[region_pool] = -10001 - c


        #topk_indices_flat = (top_non_overlapping_idx + torch.arange(0,orig_score.shape[0]*orig_score.shape[1],orig_score.shape[1],device=orig_score.device).unsqueeze(-1)).view(-1)
        #top_k_non_overlapping = orig_score.view(-1).index_select(0,topk_indices_flat).view(top_non_overlapping_idx.shape[0],-1)
        #top_k_non_overlapping[top_k_non_overlapping <= -9900] = 0
        #score = (top_k_non_overlapping * self.chunk_scoring).sum(dim=1)
       
        top_non_overlapping_idx_neighbors = torch.cat([top_non_overlapping_idx,top_non_overlapping_idx - 1,top_non_overlapping_idx - 2,top_non_overlapping_idx + 1,top_non_overlapping_idx + 2],dim=1)
        top_non_overlapping_idx_neighbors[top_non_overlapping_idx_neighbors < 0] = 0
        top_non_overlapping_idx_neighbors[top_non_overlapping_idx_neighbors >= orig_score.shape[1]] = orig_score.shape[1] - 1
#
        topk_indices_flat = (top_non_overlapping_idx_neighbors + torch.arange(0,orig_score.shape[0]*orig_score.shape[1],orig_score.shape[1],device=orig_score.device).unsqueeze(-1)).view(-1)
        top_k_non_overlapping = orig_score.view(-1).index_select(0,topk_indices_flat).view(top_non_overlapping_idx.shape[0],-1)
        top_k_non_overlapping[top_k_non_overlapping <= -9900] = 0
#
        score = (top_k_non_overlapping * self.chunk_scoring).sum(dim=1)

        if output_secondary_output:
            query_mean_vector = query_embeddings.sum(dim=1) / query_pad_oov_mask.sum(dim=1).unsqueeze(-1)
            #sat_influence_from_top_k = sat_influencer.transpose(1,2).reshape(-1,query_embeddings.shape[1],2).index_select(0,topk_indices_flat).view(top_non_overlapping_idx_neighbors.shape[0],top_non_overlapping_idx_neighbors.shape[1],query_embeddings.shape[1],2)
            return score, {"score":score,"orig_score":orig_score,"top_non_overlapping_idx":top_non_overlapping_idx,"orig_doc_len":orig_doc_len,"top_k_non_overlapping":top_k_non_overlapping,#"sat_influence_from_top_k":sat_influence_from_top_k,
                           #"total_chunks":chunked_docs2.shape[0],"packed_chunks":documents_packed.shape[0]
                           }
                           #"query_mean_vector":query_mean_vector,"cosine_matrix_masked":cosine_matrix}
        else:
            return score

    def forward_representation(self, sequence_embeddings: torch.Tensor, sequence_mask: torch.Tensor, positional_features=None) -> torch.Tensor:

        pos_sequence = sequence_embeddings
        if self.use_pos_encoding:
            if positional_features is None:
                positional_features = self.positional_features_d[:,:sequence_embeddings.shape[1],:]
            pos_sequence = sequence_embeddings + positional_features
        
        sequence_embeddings_context = self.contextualizer((pos_sequence).transpose(1,0),src_key_padding_mask=~sequence_mask.bool()).transpose(1,0)
        
        sequence_embeddings = (self.mixer * sequence_embeddings + (1 - self.mixer) * sequence_embeddings_context) * sequence_mask.unsqueeze(-1)

        return sequence_embeddings

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
        return "TK: dense w: "+str(self.dense.weight.data) +\
        " self.chunk_scoring: " +str(self.chunk_scoring.data) +\
        " self.saturation_linear: " +str(self.saturation_linear.weight.data) + " bias: " +str(self.saturation_linear.bias.data) +\
        " self.saturation_linear2: " +str(self.saturation_linear2.weight.data) + " bias: " +str(self.saturation_linear2.bias.data) +\
        " self.saturation_linear3: " +str(self.saturation_linear3.weight.data) + " bias: " +str(self.saturation_linear3.bias.data) +\
        "mixer: "+str(self.mixer.data)

    def get_param_secondary(self):
        return {"dense_weight":self.dense.weight,
                "saturation_linear_weight":self.saturation_linear.weight,
                "saturation_linear_bias":self.saturation_linear.bias,
                "saturation_linear2_weight":self.saturation_linear2.weight,
                "saturation_linear2_bias":self.saturation_linear2.bias,
                "saturation_linear3_weight":self.saturation_linear3.weight,
                "saturation_linear3_bias":self.saturation_linear3.bias,
                "chunk_scoring":self.chunk_scoring,
                "mixer":self.mixer}