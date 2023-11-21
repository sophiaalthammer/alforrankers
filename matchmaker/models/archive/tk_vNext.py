from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          
from allennlp.modules.matrix_attention.dot_product_matrix_attention import *                          
import math

class TK_vNext_1(nn.Module):
    '''
    TK is a neural IR model - a fusion between transformer contextualization & kernel-based scoring

    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return TK_vNext_1(word_embeddings_out_dim, 
                     kernels_mu =    config["tk_kernels_mu"],
                     kernels_sigma = config["tk_kernels_sigma"],
                     att_heads =     config["tk_att_heads"],
                     att_layer =     config["tk_att_layer"],
                     att_proj_dim =  config["tk_att_proj_dim"],
                     att_ff_dim =    config["tk_att_ff_dim"],
                     max_length =    config["max_doc_length"],
                     use_pos_encoding     = config["tk_use_pos_encoding"],
                     use_diff_posencoding = config["tk_use_diff_posencoding"],
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
                 #position_bias_bin_percent,
                 #position_bias_absolute_steps
                 ):

        super(TK_vNext_1, self).__init__()

        n_kernels = len(kernels_mu)
        self.use_pos_encoding     = use_pos_encoding    
        self.use_diff_posencoding = use_diff_posencoding

        if len(kernels_mu) != len(kernels_sigma):
            raise Exception("len(kernels_mu) != len(kernels_sigma)")

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(kernels_mu), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(kernels_sigma), requires_grad=False).view(1, 1, 1, n_kernels)


        pos_f = self.get_positional_features(_embsize,max_length) #max_timescale=100000
        pos_f.requires_grad = False
        self.positional_features_q = pos_f #nn.Parameter(pos_f)
        self.positional_features_q.requires_grad = False

        if self.use_diff_posencoding == True:
            pos_f = self.get_positional_features(_embsize,max_length+500) #max_timescale=100000
            pos_f.requires_grad = False
            self.positional_features_d = pos_f[:,500:,:] #nn.Parameter(pos_f)
            self.positional_features_d.requires_grad = False
        else:
            self.positional_features_d = self.positional_features_q


        self.mixer = nn.Parameter(torch.full([1], 0.5, dtype=torch.float32, requires_grad=True))

        self.emb_reducer = nn.Linear(_embsize, 100, bias=True)


        encoder_layer = nn.TransformerEncoderLayer(100, att_heads, dim_feedforward=att_ff_dim, dropout=0)
        self.contextualizer = nn.TransformerEncoder(encoder_layer, att_layer, norm=None)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        self.dense = nn.Linear(n_kernels, 1, bias=False)
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014) # inits taken from matchzoo

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor, 
                output_secondary_output: bool = False) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # contextualization
        # -------------------------------------------------------

        query_embeddings = self.forward_representation(self.emb_reducer(query_embeddings), query_pad_oov_mask,self.positional_features_q[:,:query_embeddings.shape[1],:])
        document_embeddings = self.forward_representation(self.emb_reducer(document_embeddings), document_pad_oov_mask,self.positional_features_d[:,:document_embeddings.shape[1],:])

        #
        # masks 
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)
        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        #
        # cosine matrix
        # -------------------------------------------------------


        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------

        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view


        #
        # kernel-pooling
        # -------------------------------------------------------

        per_kernel_query = torch.sum(kernel_results_masked, 2) 

        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10))
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 


        ##
        ## weight kernel bins
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        score = torch.squeeze(dense_out,1)

        if output_secondary_output:
            query_mean_vector = query_embeddings.sum(dim=1) / query_pad_oov_mask.sum(dim=1).unsqueeze(-1)
            return score, {"score":score,"dense_out":dense_out,"per_kernel":per_kernel, # "dense_mean_out":dense_mean_out,"per_kernel_mean":per_kernel_mean,
                           "query_mean_vector":query_mean_vector,"cosine_matrix_masked":cosine_matrix_masked}
        else:
            return score

    def forward_representation(self, sequence_embeddings: torch.Tensor, sequence_mask: torch.Tensor,positional_features=None) -> torch.Tensor:

        if positional_features is None:
            positional_features = self.positional_features_d[:,:sequence_embeddings.shape[1],:]

        sequence_embeddings = sequence_embeddings * sequence_mask.unsqueeze(-1)

        pos_sequence = sequence_embeddings
        if self.use_pos_encoding:
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

    

    def get_param_stats(self): #" b: "+str(self.dense.bias.data) +\ "b: "+str(self.dense_mean.bias.data) +#"scaler: "+str(self.nn_scaler.data) +\
        return "TK: dense w: "+str(self.dense.weight.data) +\
        "mixer: "+str(self.mixer.data)

    def get_param_secondary(self):
        return {"dense_weight":self.dense.weight,#"dense_bias":self.dense.bias,
                #"dense_mean_weight":self.dense_mean.weight,#"dense_mean_bias":self.dense_mean.bias,
                #"dense_comb_weight":self.dense_comb.weight, 
                #"scaler":self.nn_scaler ,
                "mixer":self.mixer}


class TK_vNext_2(nn.Module):
    '''
    TK is a neural IR model - a fusion between transformer contextualization & kernel-based scoring

    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return TK_vNext_2(word_embeddings_out_dim, 
                     kernels_mu =    config["tk_kernels_mu"],
                     kernels_sigma = config["tk_kernels_sigma"],
                     att_heads =     config["tk_att_heads"],
                     att_layer =     config["tk_att_layer"],
                     att_proj_dim =  config["tk_att_proj_dim"],
                     att_ff_dim =    config["tk_att_ff_dim"],
                     max_length =    config["max_doc_length"],
                     use_pos_encoding     = config["tk_use_pos_encoding"],
                     use_diff_posencoding = config["tk_use_diff_posencoding"],
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
                 #position_bias_bin_percent,
                 #position_bias_absolute_steps
                 ):

        super(TK_vNext_2, self).__init__()

        n_kernels = len(kernels_mu)
        self.use_pos_encoding     = use_pos_encoding    
        self.use_diff_posencoding = use_diff_posencoding

        if len(kernels_mu) != len(kernels_sigma):
            raise Exception("len(kernels_mu) != len(kernels_sigma)")

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(kernels_mu), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(kernels_sigma), requires_grad=False).view(1, 1, 1, n_kernels)


        pos_f = self.get_positional_features(_embsize,max_length) #max_timescale=100000
        pos_f.requires_grad = False
        self.positional_features_q = pos_f #nn.Parameter(pos_f)
        self.positional_features_q.requires_grad = False

        if self.use_diff_posencoding == True:
            pos_f = self.get_positional_features(_embsize,max_length+500) #max_timescale=100000
            pos_f.requires_grad = False
            self.positional_features_d = pos_f[:,500:,:] #nn.Parameter(pos_f)
            self.positional_features_d.requires_grad = False
        else:
            self.positional_features_d = self.positional_features_q


        self.mixer = nn.Parameter(torch.full([1], 0.5, dtype=torch.float32, requires_grad=True))

        self.emb_reducer = nn.Linear(_embsize, 100, bias=True)


        encoder_layer = nn.TransformerEncoderLayer(100, att_heads, dim_feedforward=att_ff_dim, dropout=0)
        self.contextualizer = nn.TransformerEncoder(encoder_layer, att_layer, norm=None)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        self.dense = nn.Linear(n_kernels, 1, bias=False)
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014) # inits taken from matchzoo

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor, 
                output_secondary_output: bool = False) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # contextualization
        # -------------------------------------------------------

        query_embeddings = self.forward_representation(self.emb_reducer(query_embeddings), query_pad_oov_mask,self.positional_features_q[:,:query_embeddings.shape[1],:])

        chunk_size = 24
        overlap = 8
        document_embeddings = self.emb_reducer(document_embeddings) * document_pad_oov_mask.unsqueeze(-1)

        document_embeddings = nn.functional.pad(document_embeddings,(0,0,0, chunk_size -(document_embeddings.shape[1] % chunk_size)))
        document_pad_oov_mask = nn.functional.pad(document_pad_oov_mask,(0, chunk_size -(document_pad_oov_mask.shape[1] % chunk_size)))

        chunked_docs = document_embeddings.unfold(1,chunk_size,chunk_size).transpose(-1,-2)
        chunked_pad= document_pad_oov_mask.unfold(1,chunk_size,chunk_size)#.transpose(-1,-2)
        
        batch_size = chunked_docs.shape[0]
        chunk_pieces = chunked_docs.shape[1]

        chunked_docs2=chunked_docs.reshape(-1,chunk_size,document_embeddings.shape[-1])
        chunked_pad2=chunked_pad.reshape(-1,chunk_size)
        #chunked_pad2[chunked_pad2.sum(-1) == 0] = 1

        documents_nopad = chunked_docs2[chunked_pad2.sum(-1) != 0]
        pad_nopad = chunked_pad2[chunked_pad2.sum(-1) != 0]

        #document_embeddings = self.forward_representation(chunked_docs2, chunked_pad2, self.positional_features_d[:,:document_embeddings.shape[1],:])
        documents_nopad = self.forward_representation(documents_nopad, pad_nopad, self.positional_features_d[:,:document_embeddings.shape[1],:])

        empty_test = torch.zeros_like(chunked_docs2)

        empty_test[chunked_pad2.sum(-1) != 0] = documents_nopad

        #document_embeddings[chunked_pad2.sum(-1) == 0,:] = 0 
        #document_embeddings[document_embeddings != document_embeddings] = 0

        document_embeddings = empty_test.view(batch_size,-1,chunk_size,document_embeddings.shape[-1]).view(batch_size,-1,document_embeddings.shape[-1])


        #assert not torch.isnan(document_embeddings).any()

        #
        # masks 
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)
        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        #
        # cosine matrix
        # -------------------------------------------------------


        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------

        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view


        #
        # kernel-pooling
        # -------------------------------------------------------

        per_kernel_query = torch.sum(kernel_results_masked, 2) 

        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10))
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 


        ##
        ## weight kernel bins
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        score = torch.squeeze(dense_out,1)

        if output_secondary_output:
            query_mean_vector = query_embeddings.sum(dim=1) / query_pad_oov_mask.sum(dim=1).unsqueeze(-1)
            return score, {"score":score,"dense_out":dense_out,"per_kernel":per_kernel, # "dense_mean_out":dense_mean_out,"per_kernel_mean":per_kernel_mean,
                           "query_mean_vector":query_mean_vector,"cosine_matrix_masked":cosine_matrix_masked}
        else:
            return score

    def forward_representation(self, sequence_embeddings: torch.Tensor, sequence_mask: torch.Tensor,positional_features=None) -> torch.Tensor:

        if positional_features is None:
            positional_features = self.positional_features_d[:,:sequence_embeddings.shape[1],:]

        sequence_embeddings = sequence_embeddings * sequence_mask.unsqueeze(-1)

        pos_sequence = sequence_embeddings
        if self.use_pos_encoding:
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

    

    def get_param_stats(self): #" b: "+str(self.dense.bias.data) +\ "b: "+str(self.dense_mean.bias.data) +#"scaler: "+str(self.nn_scaler.data) +\
        return "TK: dense w: "+str(self.dense.weight.data) +\
        "mixer: "+str(self.mixer.data)

    def get_param_secondary(self):
        return {"dense_weight":self.dense.weight,#"dense_bias":self.dense.bias,
                #"dense_mean_weight":self.dense_mean.weight,#"dense_mean_bias":self.dense_mean.bias,
                #"dense_comb_weight":self.dense_comb.weight, 
                #"scaler":self.nn_scaler ,
                "mixer":self.mixer}

class TK_vNext_2_windows(nn.Module):
    '''
    TK is a neural IR model - a fusion between transformer contextualization & kernel-based scoring

    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return TK_vNext_2_windows(word_embeddings_out_dim, 
                     kernels_mu =    config["tk_kernels_mu"],
                     kernels_sigma = config["tk_kernels_sigma"],
                     att_heads =     config["tk_att_heads"],
                     att_layer =     config["tk_att_layer"],
                     att_proj_dim =  config["tk_att_proj_dim"],
                     att_ff_dim =    config["tk_att_ff_dim"],
                     max_length =    config["max_doc_length"],
                     use_pos_encoding     = config["tk_use_pos_encoding"],
                     use_diff_posencoding = config["tk_use_diff_posencoding"],
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
                 #position_bias_bin_percent,
                 #position_bias_absolute_steps
                 ):

        super(TK_vNext_2_windows, self).__init__()

        n_kernels = len(kernels_mu)
        self.use_pos_encoding     = use_pos_encoding    
        self.use_diff_posencoding = use_diff_posencoding

        if len(kernels_mu) != len(kernels_sigma):
            raise Exception("len(kernels_mu) != len(kernels_sigma)")

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(kernels_mu), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(kernels_sigma), requires_grad=False).view(1, 1, 1, n_kernels)


        pos_f = self.get_positional_features(_embsize,max_length) #max_timescale=100000
        pos_f.requires_grad = False
        self.positional_features_q = pos_f #nn.Parameter(pos_f)
        self.positional_features_q.requires_grad = False

        if self.use_diff_posencoding == True:
            pos_f = self.get_positional_features(_embsize,max_length+500) #max_timescale=100000
            pos_f.requires_grad = False
            self.positional_features_d = pos_f[:,500:,:] #nn.Parameter(pos_f)
            self.positional_features_d.requires_grad = False
        else:
            self.positional_features_d = self.positional_features_q


        self.mixer = nn.Parameter(torch.full([1], 0.5, dtype=torch.float32, requires_grad=True))

        self.emb_reducer = nn.Linear(_embsize, 100, bias=True)


        encoder_layer = nn.TransformerEncoderLayer(_embsize, att_heads, dim_feedforward=att_ff_dim, dropout=0)
        self.contextualizer = nn.TransformerEncoder(encoder_layer, att_layer, norm=None)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        #self.dense = nn.Linear(n_kernels, 1, bias=False)
        #torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014) # inits taken from matchzoo

        win_size= [20,30,50,80,100,120,150]
        max_windows = [math.ceil(max_length / float(w)) for w in win_size]


        self.kernel_weights = nn.ModuleList([nn.Linear(n_kernels, 1, bias=False) for w in win_size])
        self.nn_scaler = nn.ParameterList([nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True)) for w in win_size])

        self.window_size = win_size
        self.window_scorer = []
        for w in max_windows:
            l =  nn.Linear(w, 1, bias=False)
            torch.nn.init.constant_(l.weight, 1/w)
            self.window_scorer.append(l)

        self.window_scorer = nn.ModuleList(self.window_scorer)

        self.window_merger = nn.Linear(len(self.window_size), 1, bias=False)


    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor, 
                output_secondary_output: bool = False) -> torch.Tensor:
        
        #query_embeddings = self.emb_reducer(query_embeddings)
        if self.use_pos_encoding:
            query_embeddings = query_embeddings + self.positional_features_q[:,:query_embeddings.shape[1],:]

        query_embeddings = self.forward_representation(query_embeddings, query_pad_oov_mask)

        chunk_size = 40
        overlap = 5

        extended_chunk_size = chunk_size + 2 * overlap
        needed_padding = extended_chunk_size - ((document_pad_oov_mask.shape[1]-overlap) % chunk_size)
        orig_doc_len = document_pad_oov_mask.shape[1]
        #x = torch.range(1,106)
        #x = nn.functional.pad(x,(overlap,needed_padding))
        #unique_entries = x.unfold(0,extended_chunk_size,chunk_size)[:,overlap:-overlap]

        #document_embeddings = self.emb_reducer(document_embeddings)
        #if self.use_pos_encoding:
        #    document_embeddings = document_embeddings+ self.positional_features_d[:,:document_embeddings.shape[1],:]

        document_embeddings = nn.functional.pad(document_embeddings,(0,0,overlap, needed_padding))
        document_pad_oov_mask = nn.functional.pad(document_pad_oov_mask,(overlap, needed_padding))

        chunked_docs = document_embeddings.unfold(1,extended_chunk_size,chunk_size).transpose(-1,-2)#[:,:,overlap:-overlap,:]
        chunked_pad = document_pad_oov_mask.unfold(1,extended_chunk_size,chunk_size)#[:,:,overlap:-overlap]
        
        batch_size = chunked_docs.shape[0]
        chunk_pieces = chunked_docs.shape[1]

        chunked_docs2=chunked_docs.reshape(-1,extended_chunk_size,document_embeddings.shape[-1])
        chunked_pad2=chunked_pad.reshape(-1,extended_chunk_size)
        #chunked_pad2[chunked_pad2.sum(-1) == 0] = 1

        packed_indices = chunked_pad2[:,overlap:-overlap].sum(-1) != 0

        documents_packed = chunked_docs2[packed_indices]
        padding_packed = chunked_pad2[packed_indices]

        if self.use_pos_encoding:
            documents_packed = documents_packed + self.positional_features_d[:,:documents_packed.shape[1],:]


        documents_packed = self.forward_representation(documents_packed, padding_packed)

        documents_unique_again = documents_packed[:,overlap:-overlap,:]
        #document_mask_unique_again = chunked_pad[:,:,overlap:-overlap]
        document_mask_packed_unique = padding_packed[:,overlap:-overlap]
        #        
        # reshape back in original form
 
        unpacked_documents = torch.zeros((chunked_docs2.shape[0],documents_unique_again.shape[1],chunked_docs2.shape[2]), dtype=chunked_docs2.dtype, layout=chunked_docs2.layout, device=chunked_docs2.device)
        unpacked_documents[packed_indices] = documents_unique_again

        document_embeddings = unpacked_documents.view(batch_size,-1,chunk_size,document_embeddings.shape[-1]).view(batch_size,-1,document_embeddings.shape[-1])
        updated_mask = chunked_pad[:,:,overlap:-overlap].view(batch_size,-1)

        document_pad_oov_mask = updated_mask

        #
        # masks 
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)
        #doc_lengths = torch.sum(document_pad_oov_mask, 1)

        #
        # cosine matrix
        # -------------------------------------------------------


        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------

        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view


        #
        # kernel-pooling
        # -------------------------------------------------------

        individual_window_scores = []

        for i,window in enumerate(self.window_size):

            kernel_results_masked = nn.functional.pad(kernel_results_masked,(0,0,0,window - kernel_results_masked.shape[-2]%window)) 

            scoring_windows = kernel_results_masked.unfold(dimension=-2,size=window,step=window)

            scoring_windows = scoring_windows.transpose(-1,-2)
            #kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

            per_kernel_query = torch.sum(scoring_windows, -2)
            log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) #* 
            log_per_kernel_query_masked = log_per_kernel_query * (per_kernel_query.sum(dim=-1) != 0).unsqueeze(-1).float()
            #log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1).unsqueeze(-1) # make sure we mask out padding values
            per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

            window_scores = self.kernel_weights[i](per_kernel).squeeze(-1)

            window_scores_exp = torch.exp(window_scores * self.nn_scaler[i]) * (window_scores != 0).float()
            #window_scores_exp=window_scores
            if window_scores_exp.shape[-1] > self.window_scorer[i].in_features:
                window_scores_exp = window_scores_exp[:,:self.window_scorer[i].in_features]
            if window_scores_exp.shape[-1] < self.window_scorer[i].in_features:
                window_scores_exp = nn.functional.pad(window_scores_exp,(0,self.window_scorer[i].in_features - window_scores_exp.shape[-1])) 
#   
            window_scores_exp = window_scores_exp.sort(dim=-1, descending=True)[0]

            individual_window_scores.append(self.window_scorer[i](window_scores_exp))
        #final_score = window_scores.sum(dim=-1) / (window_scores != 0).sum(dim=-1).float()

        

        final_window_score = self.window_merger(torch.cat(individual_window_scores,dim=1))
        score = torch.squeeze(final_window_score,1) #torch.tanh(dense_out), 1)
        if output_secondary_output:
            return score, {}
        return score

    def forward_representation(self, sequence_embeddings: torch.Tensor, sequence_mask: torch.Tensor,positional_features=None) -> torch.Tensor:

        #if positional_features is None:
        #    positional_features = self.positional_features_d[:,:sequence_embeddings.shape[1],:]

        sequence_embeddings = sequence_embeddings * sequence_mask.unsqueeze(-1)

        #pos_sequence = sequence_embeddings
        #if self.use_pos_encoding:
        #    pos_sequence = sequence_embeddings + positional_features
        
        sequence_embeddings_context = self.contextualizer((sequence_embeddings).transpose(1,0),src_key_padding_mask=~sequence_mask.bool()).transpose(1,0)
        
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

    

    def get_param_stats(self):
        return "tk_v2: "+\
            " ".join([" kernel_weight ("+str(self.window_size[i])+")"+str(w.weight.data) for i,w in enumerate(self.kernel_weights)])+"\n"+\
            " ".join([" nn_scaler ("+str(self.window_size[i])+")"+str(w.data) for i,w in enumerate(self.nn_scaler)])+"\n"+\
            " ".join([" window_scorer ("+str(self.window_size[i])+")"+str(w.weight.data) for i,w in enumerate(self.window_scorer)])+"\n"+\
            "mixer: "+str(self.mixer.data) + "window_merger: "+str(self.window_merger.weight.data)

    def get_param_secondary(self):
        return {#"dense_weight":self.dense.weight,"dense_bias":self.dense.bias,
                #"dense_mean_weight":self.dense_mean.weight,"dense_mean_bias":self.dense_mean.bias,
                "window_merger":self.window_merger.weight, 
                #"scaler: ":self.nn_scaler ,
                "mixer: ":self.mixer}

from matchmaker.modules.masked_softmax import MaskedSoftmax

class TK_vNext_3(nn.Module):
    '''
    TK is a neural IR model - a fusion between transformer contextualization & kernel-based scoring

    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return TK_vNext_3(word_embeddings_out_dim, 
                     kernels_mu =    config["tk_kernels_mu"],
                     kernels_sigma = config["tk_kernels_sigma"],
                     att_heads =     config["tk_att_heads"],
                     att_layer =     config["tk_att_layer"],
                     att_proj_dim =  config["tk_att_proj_dim"],
                     att_ff_dim =    config["tk_att_ff_dim"],
                     max_length =    config["max_doc_length"],
                     use_pos_encoding     = config["tk_use_pos_encoding"],
                     use_diff_posencoding = config["tk_use_diff_posencoding"],
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
                 #position_bias_bin_percent,
                 #position_bias_absolute_steps
                 ):

        super(TK_vNext_3, self).__init__()

        n_kernels = len(kernels_mu)
        self.use_pos_encoding     = use_pos_encoding    
        self.use_diff_posencoding = use_diff_posencoding

        if len(kernels_mu) != len(kernels_sigma):
            raise Exception("len(kernels_mu) != len(kernels_sigma)")

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(kernels_mu), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(kernels_sigma), requires_grad=False).view(1, 1, 1, n_kernels)


        pos_f = self.get_positional_features(_embsize,max_length) #max_timescale=100000
        pos_f.requires_grad = False
        self.positional_features_q = pos_f #nn.Parameter(pos_f)
        self.positional_features_q.requires_grad = False

        if self.use_diff_posencoding == True:
            pos_f = self.get_positional_features(_embsize,max_length+500) #max_timescale=100000
            pos_f.requires_grad = False
            self.positional_features_d = pos_f[:,500:,:] #nn.Parameter(pos_f)
            self.positional_features_d.requires_grad = False
        else:
            self.positional_features_d = self.positional_features_q


        self.mixer = nn.Parameter(torch.full([1], 0.5, dtype=torch.float32, requires_grad=True))

        self.emb_reducer = nn.Linear(_embsize, 100, bias=True)


        encoder_layer = nn.TransformerEncoderLayer(_embsize, att_heads, dim_feedforward=att_ff_dim, dropout=0)
        self.contextualizer = nn.TransformerEncoder(encoder_layer, att_layer, norm=None)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()


        self.kernel_bias = nn.Parameter(torch.full([1,1,n_kernels], 0.5, dtype=torch.float32, requires_grad=True))
        self.kernel_mult = nn.Parameter(torch.full([1,1,n_kernels], 1, dtype=torch.float32, requires_grad=True))
        
        
        self.chunk_scoring = nn.Parameter(torch.full([1,3], 1, dtype=torch.float32, requires_grad=True))


        self.dense = nn.Linear(n_kernels, 1, bias=False)
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014) # inits taken from matchzoo

        self.masked_softmax = MaskedSoftmax()

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor, 
                output_secondary_output: bool = False) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # contextualization
        # -------------------------------------------------------

        #query_embeddings = self.emb_reducer(query_embeddings)
        if self.use_pos_encoding:
            query_embeddings = query_embeddings + self.positional_features_q[:,:query_embeddings.shape[1],:]

        query_embeddings = self.forward_representation(query_embeddings, query_pad_oov_mask)

        chunk_size = 40
        overlap = 5

        extended_chunk_size = chunk_size + 2 * overlap
        needed_padding = extended_chunk_size - ((document_pad_oov_mask.shape[1]-overlap) % chunk_size)
        orig_doc_len = document_pad_oov_mask.shape[1]
        #x = torch.range(1,106)
        #x = nn.functional.pad(x,(overlap,needed_padding))
        #unique_entries = x.unfold(0,extended_chunk_size,chunk_size)[:,overlap:-overlap]

        #document_embeddings = self.emb_reducer(document_embeddings)
        #if self.use_pos_encoding:
        #    document_embeddings = document_embeddings+ self.positional_features_d[:,:document_embeddings.shape[1],:]

        document_embeddings = nn.functional.pad(document_embeddings,(0,0,overlap, needed_padding))
        document_pad_oov_mask = nn.functional.pad(document_pad_oov_mask,(overlap, needed_padding))

        chunked_docs = document_embeddings.unfold(1,extended_chunk_size,chunk_size).transpose(-1,-2)#[:,:,overlap:-overlap,:]
        chunked_pad = document_pad_oov_mask.unfold(1,extended_chunk_size,chunk_size)#[:,:,overlap:-overlap]
        
        batch_size = chunked_docs.shape[0]
        chunk_pieces = chunked_docs.shape[1]

        chunked_docs2=chunked_docs.reshape(-1,extended_chunk_size,document_embeddings.shape[-1])
        chunked_pad2=chunked_pad.reshape(-1,extended_chunk_size)
        #chunked_pad2[chunked_pad2.sum(-1) == 0] = 1

        packed_indices = chunked_pad2[:,overlap:-overlap].sum(-1) != 0

        documents_packed = chunked_docs2[packed_indices]
        padding_packed = chunked_pad2[packed_indices]

        if self.use_pos_encoding:
            documents_packed = documents_packed + self.positional_features_d[:,:documents_packed.shape[1],:]


        documents_packed = self.forward_representation(documents_packed, padding_packed)

        documents_unique_again = documents_packed[:,overlap:-overlap,:]
        #document_mask_unique_again = chunked_pad[:,:,overlap:-overlap]
        document_mask_packed_unique = padding_packed[:,overlap:-overlap]
        #        
        # reshape back in original form
 
        #unpacked_documents = torch.zeros((chunked_docs2.shape[0],documents_unique_again.shape[1],chunked_docs2.shape[2]), dtype=chunked_docs2.dtype, layout=chunked_docs2.layout, device=chunked_docs2.device)
        #unpacked_documents[packed_indices] = documents_unique_again
#
        #document_embeddings = unpacked_documents.view(batch_size,-1,chunk_size,document_embeddings.shape[-1]).view(batch_size,-1,document_embeddings.shape[-1])
        #updated_mask = chunked_pad[:,:,overlap:-overlap].view(batch_size,-1)
#
        #document_pad_oov_mask = updated_mask
        #if (padding_packed.sum(dim=1) < 2).any():
        #    print("error incoming")
        #    for i in range(0,padding_packed.shape[0]):
        #        if padding_packed[i].sum() < 0.5:
        #            print(padding_packed[i])
        #            print(torch.isnan(documents_packed[i]),documents_packed[i])
#
        #if orig_doc_len ==106:
        #    print("error incoming")
        #    for i in range(0,document_mask_packed_unique.shape[0]):
        #        print(document_mask_packed_unique[i])
        #        print(documents_packed[i])
        
            #assert False
        #if torch.isnan(documents_packed).any():
        #    for i in range(0,padding_packed.shape[0]):
        #        print(padding_packed[i])
        #        print(torch.isnan(documents_packed[i]),documents_packed[i])
        #print(orig_doc_len,needed_padding)
        assert not torch.isnan(documents_packed).any()

        #
        # masks 
        # -------------------------------------------------------

        #query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        #query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)
        #doc_lengths = torch.sum(document_pad_oov_mask, 1)

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


        #
        # kernel-pooling
        # -------------------------------------------------------

        per_kernel_query = torch.sum(kernel_results_masked, 2) 

        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query * self.kernel_mult , min=1e-10)) #self.kernel_bias
        log_per_kernel_query_masked = log_per_kernel_query * packed_query_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 


        ##
        ## weight kernel bins
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)

        # need to combine scores again
        all_scores = torch.zeros((chunked_docs2.shape[0],1), dtype=chunked_docs2.dtype, layout=chunked_docs2.layout, device=chunked_docs2.device)
        all_scores[packed_indices] = dense_out

        # score per doc & chunk !
        score = all_scores.view(batch_size,chunk_pieces,1).squeeze(-1)

        #self.chunk_scoring
        top_k = 3

        if score.shape[1] < top_k:
            score = nn.functional.pad(score,(0, top_k - score.shape[1]))

        score[score == 0] = -1000
        score = score.topk(k=top_k,dim=1)[0]
        score[score == -1000] = 0
        
        score = (score*self.chunk_scoring).sum(dim=1)

        #score = (score * self.masked_softmax(score, score != 0)).sum(dim=1)
        #score = score.max(dim=1)[0]

        if output_secondary_output:
            query_mean_vector = query_embeddings.sum(dim=1) / query_pad_oov_mask.sum(dim=1).unsqueeze(-1)
            return score, {"score":score,"dense_out":dense_out,"per_kernel":per_kernel, "total_chunks":chunked_docs2.shape[0],"packed_chunks":dense_out.shape[0],
                           "query_mean_vector":query_mean_vector,"cosine_matrix_masked":cosine_matrix}
        else:
            return score

    def forward_representation(self, sequence_embeddings: torch.Tensor, sequence_mask: torch.Tensor,positional_features=None) -> torch.Tensor:

        #if positional_features is None:
        #    positional_features = self.positional_features_d[:,:sequence_embeddings.shape[1],:]

        sequence_embeddings = sequence_embeddings * sequence_mask.unsqueeze(-1)

        #pos_sequence = sequence_embeddings
        #if self.use_pos_encoding:
        #    pos_sequence = sequence_embeddings + positional_features
        
        sequence_embeddings_context = self.contextualizer((sequence_embeddings).transpose(1,0),src_key_padding_mask=~sequence_mask.bool()).transpose(1,0)
        
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

    

    def get_param_stats(self): #" b: "+str(self.dense.bias.data) +\ "b: "+str(self.dense_mean.bias.data) +#"scaler: "+str(self.nn_scaler.data) +\
        return "TK: dense w: "+str(self.dense.weight.data) +\
        "self.kernel_mult:" +str(self.kernel_mult.data)+ " self.chunk_scoring: " +str(self.chunk_scoring.data) +\
        "mixer: "+str(self.mixer.data)

    def get_param_secondary(self):
        return {"dense_weight":self.dense.weight,#"dense_bias":self.dense.bias,
                #"dense_mean_weight":self.dense_mean.weight,#"dense_mean_bias":self.dense_mean.bias,
                #"dense_comb_weight":self.dense_comb.weight, 
                #"scaler":self.nn_scaler ,
                "mixer":self.mixer}

from matchmaker.modules.neuralIR_encoder import get_vectors_n_masks
from matchmaker.modules.bert_parts import BertEncoderReduced

class TK_vNext_4(nn.Module):

    '''
    TK is a neural IR model - a fusion between transformer contextualization & kernel-based scoring

    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions

    '''

    @staticmethod
    def from_config(config,word_embeddings_layer,bert_layers):
        if config["bert_emb_layers"] != -1:
            reduced_layers = bert_layers[0].layer[:config["bert_emb_layers"]]
            bert_pos = bert_layers[1]
            bert_layers = BertEncoderReduced(reduced_layers)

        return TK_vNext_4(word_embeddings_layer, bert_layers,bert_pos,
                     kernels_mu =    config["tk_kernels_mu"],
                     kernels_sigma = config["tk_kernels_sigma"],
                     att_heads =     config["tk_att_heads"],
                     att_layer =     config["tk_att_layer"],
                     att_proj_dim =  config["tk_att_proj_dim"],
                     att_ff_dim =    config["tk_att_ff_dim"],
                     max_length =    config["max_doc_length"],
                     use_pos_encoding     = config["tk_use_pos_encoding"],
                     use_diff_posencoding = config["tk_use_diff_posencoding"],
                     #position_bias_bin_percent   = config["tk_position_bias_bin_percent"],
                     #position_bias_absolute_steps= config["tk_position_bias_absolute_steps"] 
                     )

    def __init__(self,
                 word_embeddings_layer, bert_layers,bert_pos,
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
                 #position_bias_bin_percent,
                 #position_bias_absolute_steps
                 ):

        super(TK_vNext_4, self).__init__()

        _embsize = word_embeddings_layer.get_output_dim()
        self.word_embeddings_layer = word_embeddings_layer
        self.bert_layers = bert_layers
        self.bert_pos_emb = bert_pos

        n_kernels = len(kernels_mu)
        self.use_pos_encoding     = use_pos_encoding    
        self.use_diff_posencoding = use_diff_posencoding

        if len(kernels_mu) != len(kernels_sigma):
            raise Exception("len(kernels_mu) != len(kernels_sigma)")

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(kernels_mu), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(kernels_sigma), requires_grad=False).view(1, 1, 1, n_kernels)


        #pos_f = self.get_positional_features(_embsize,max_length) #max_timescale=100000
        #pos_f.requires_grad = False
        #self.positional_features_q = pos_f #nn.Parameter(pos_f)
        #self.positional_features_q.requires_grad = False
#
        #if self.use_diff_posencoding == True:
        #    pos_f = self.get_positional_features(_embsize,max_length+500) #max_timescale=100000
        #    pos_f.requires_grad = False
        #    self.positional_features_d = pos_f[:,500:,:] #nn.Parameter(pos_f)
        #    self.positional_features_d.requires_grad = False
        #else:
        #    self.positional_features_d = self.positional_features_q
#

        self.mixer = nn.Parameter(torch.full([1], 0.5, dtype=torch.float32, requires_grad=True))

        self.emb_reducer = nn.Linear(_embsize, 100, bias=True)


        encoder_layer = nn.TransformerEncoderLayer(100, att_heads, dim_feedforward=att_ff_dim, dropout=0)
        self.contextualizer = nn.TransformerEncoder(encoder_layer, att_layer, norm=None)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()


        self.kernel_bias = nn.Parameter(torch.full([1,1,n_kernels], 0.5, dtype=torch.float32, requires_grad=True))
        self.kernel_mult = nn.Parameter(torch.full([1,1,n_kernels], 1, dtype=torch.float32, requires_grad=True))
        
        top_k = 3
        
        self.chunk_scoring = nn.Parameter(torch.full([1,top_k], 1, dtype=torch.float32, requires_grad=True))
        self.chunk_scoring_bert = nn.Parameter(torch.full([1,top_k], 1, dtype=torch.float32, requires_grad=True))


        self.bert_score = nn.Parameter(torch.full([1], 1, dtype=torch.float32, requires_grad=True))
        self.tk_score = nn.Parameter(torch.full([1], 1, dtype=torch.float32, requires_grad=True))


        self.dense = nn.Linear(n_kernels, 1, bias=False)
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014) # inits taken from matchzoo

        #self.masked_softmax = MaskedSoftmax()

        self.bert_pooler = nn.Linear(_embsize, 1, bias=True)
        self.train_counter = 0

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor],
                output_secondary_output: bool = False) -> torch.Tensor:
        # pylint: disable=arguments-differ

        query_embeddings,document_embeddings,query_pad_oov_mask,document_pad_oov_mask = get_vectors_n_masks(self.word_embeddings_layer,query,document)


        #
        # contextualization
        # -------------------------------------------------------

        if self.use_pos_encoding:
            position_ids = torch.arange(query_embeddings.shape[1], dtype=torch.long, device=query_embeddings.device)
            position_ids = position_ids.unsqueeze(0).expand(query_embeddings.shape[0],query_embeddings.shape[1])

            query_embeddings = query_embeddings + self.bert_pos_emb(position_ids)


        query_embeddings_original = query_embeddings

        query_embeddings = self.emb_reducer(query_embeddings)
        query_embeddings = self.forward_representation(query_embeddings, query_pad_oov_mask)

        chunk_size = 60
        overlap = 5

        extended_chunk_size = chunk_size + 2 * overlap
        needed_padding = extended_chunk_size - ((document_pad_oov_mask.shape[1]-overlap) % chunk_size)
        orig_doc_len = document_pad_oov_mask.shape[1]
        #x = torch.range(1,106)
        #x = nn.functional.pad(x,(overlap,needed_padding))
        #unique_entries = x.unfold(0,extended_chunk_size,chunk_size)[:,overlap:-overlap]


        document_embeddings = nn.functional.pad(document_embeddings,(0,0,overlap, needed_padding))
        document_pad_oov_mask = nn.functional.pad(document_pad_oov_mask,(overlap, needed_padding))

        chunked_docs = document_embeddings.unfold(1,extended_chunk_size,chunk_size).transpose(-1,-2)#[:,:,overlap:-overlap,:]
        chunked_pad = document_pad_oov_mask.unfold(1,extended_chunk_size,chunk_size)#[:,:,overlap:-overlap]
        
        batch_size = chunked_docs.shape[0]
        chunk_pieces = chunked_docs.shape[1]

        chunked_docs2=chunked_docs.reshape(-1,extended_chunk_size,document_embeddings.shape[-1])
        chunked_pad2=chunked_pad.reshape(-1,extended_chunk_size)
        #chunked_pad2[chunked_pad2.sum(-1) == 0] = 1

        if self.use_pos_encoding:
            position_ids = torch.arange(30,30+chunked_docs2.shape[1], dtype=torch.long, device=chunked_docs2.device)
            position_ids = position_ids.unsqueeze(0).expand(chunked_docs2.shape[0],chunked_docs2.shape[1])

            chunked_docs2 = chunked_docs2 + self.bert_pos_emb(position_ids)

        packed_indices = chunked_pad2[:,overlap:-overlap].sum(-1) != 0

        documents_packed = chunked_docs2[packed_indices]
        padding_packed = chunked_pad2[packed_indices]

        documents_packed = self.emb_reducer(documents_packed)

        documents_packed = self.forward_representation(documents_packed, padding_packed)

        documents_unique_again = documents_packed[:,overlap:-overlap,:]
        #document_mask_unique_again = chunked_pad[:,:,overlap:-overlap]
        document_mask_packed_unique = padding_packed[:,overlap:-overlap]
        #        
        # reshape back in original form

        assert not torch.isnan(documents_packed).any()

        #
        # masks 
        # -------------------------------------------------------

        #query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        #query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)
        #doc_lengths = torch.sum(document_pad_oov_mask, 1)

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


        #
        # kernel-pooling
        # -------------------------------------------------------

        per_kernel_query = torch.sum(kernel_results_masked, 2) 

        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query * self.kernel_mult , min=1e-10)) #self.kernel_bias
        log_per_kernel_query_masked = log_per_kernel_query * packed_query_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 


        ##
        ## weight kernel bins
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)

        # need to combine scores again
        all_scores = torch.zeros((chunked_docs2.shape[0],1), dtype=chunked_docs2.dtype, layout=chunked_docs2.layout, device=chunked_docs2.device)
        all_scores[packed_indices] = dense_out

        # score per doc & chunk !
        score = all_scores.view(batch_size,chunk_pieces,1).squeeze(-1)

        #self.chunk_scoring
        top_k = 3

        if score.shape[1] < top_k:
            top_k = score.shape[1]
        #    score = nn.functional.pad(score,(0, top_k - score.shape[1]))

        score[score == 0] = -1000
        score,topk_indices = score.topk(k=top_k,dim=1)
        score[score == -1000] = 0
        
        score = (score*self.chunk_scoring[:,:top_k]).sum(dim=1).squeeze(-1)

        #score = (score * self.masked_softmax(score, score != 0)).sum(dim=1)
        #score = score.max(dim=1)[0]

        #
        # bert scoring
        #
        if not self.training or (self.training and self.train_counter >= 8):
            bert_top_k = 1
            topk_indices_flat = (topk_indices[:,0].unsqueeze(-1) + torch.arange(0,chunked_docs.shape[0]*chunked_docs.shape[1],chunked_docs.shape[1],device=chunked_docs.device).unsqueeze(-1)).view(-1)

            bert_selection = chunked_docs2.index_select(0,topk_indices_flat)
            bert_selection_mask = chunked_pad2.index_select(0,topk_indices_flat)

            bert_query_embeddings = query_embeddings_original.unsqueeze(1).expand(-1,bert_top_k,-1,-1).reshape(-1,query_embeddings_original.shape[1],query_embeddings_original.shape[-1])#[packed_indices]
            bert_query_embeddings_mask = query_pad_oov_mask.unsqueeze(1).expand(-1,bert_top_k,-1).reshape(-1,query_embeddings.shape[1])

            cls_embs = self.word_embeddings_layer.token_embedder_tokens.bert_embeddings.word_embeddings(torch.full([bert_query_embeddings.shape[0],1],101,dtype=torch.long,device=chunked_docs.device))
            sep_embs = self.word_embeddings_layer.token_embedder_tokens.bert_embeddings.word_embeddings(torch.full([bert_query_embeddings.shape[0],1],102,dtype=torch.long,device=chunked_docs.device))
            special_masks = torch.full([bert_query_embeddings.shape[0],1],1,device=chunked_docs.device)

            token_type_ids = torch.cat([torch.full([bert_query_embeddings.shape[0],bert_query_embeddings.shape[1]+1],0,dtype=torch.long,device=chunked_docs.device),
                                        torch.full([bert_query_embeddings.shape[0],bert_selection.shape[1]+1],1,dtype=torch.long,device=chunked_docs.device)],
                                        dim=1)

            token_type_embs = self.word_embeddings_layer.token_embedder_tokens.bert_embeddings.token_type_embeddings(token_type_ids)

            bert_input = torch.cat([cls_embs,bert_query_embeddings,sep_embs,bert_selection],dim=1)
            bert_input_mask = torch.cat([special_masks,bert_query_embeddings_mask,special_masks,bert_selection_mask],dim=1).unsqueeze(1).unsqueeze(2)
            bert_input_mask = (1.0 - bert_input_mask) * -10000.0

            bert_output = self.bert_layers(bert_input + token_type_embs,bert_input_mask)
            first_token_tensor = bert_output[-1][:, 0]

            bert_score = self.bert_pooler(first_token_tensor)

            #bert_selection_per_doc = bert_selection.view(topk_indices.shape[0],topk_indices.shape[1],extended_chunk_size,chunked_docs2.shape[-1])
            bert_score_per_doc = bert_score.view(topk_indices.shape[0],bert_top_k,-1).squeeze(-1)
            bert_score = bert_score_per_doc.sum(dim=1) #(bert_score_per_doc * self.chunk_scoring_bert[:,:top_k]).sum(dim=1)

            final = bert_score

        if self.training:
            #final = score * self.tk_score + bert_score * self.bert_score
            if self.train_counter < 8:
                final = score
                self.train_counter+=1
            elif self.train_counter < 16:
                final = bert_score 
                self.train_counter+=1
                if self.train_counter == 16:
                    self.train_counter=0

        if output_secondary_output:
            query_mean_vector = query_embeddings.sum(dim=1) / query_pad_oov_mask.sum(dim=1).unsqueeze(-1)
            return final, {"score":score,"dense_out":dense_out,"per_kernel":per_kernel, "total_chunks":chunked_docs2.shape[0],"packed_chunks":dense_out.shape[0],
                           "query_mean_vector":query_mean_vector,"cosine_matrix_masked":cosine_matrix}
        else:
            return final

    def forward_representation(self, sequence_embeddings: torch.Tensor, sequence_mask: torch.Tensor,positional_features=None) -> torch.Tensor:

        #if positional_features is None:
        #    positional_features = self.positional_features_d[:,:sequence_embeddings.shape[1],:]

        sequence_embeddings = sequence_embeddings * sequence_mask.unsqueeze(-1)

        #pos_sequence = sequence_embeddings
        #if self.use_pos_encoding:
        #    pos_sequence = sequence_embeddings + positional_features
        
        sequence_embeddings_context = self.contextualizer((sequence_embeddings).transpose(1,0),src_key_padding_mask=~sequence_mask.bool()).transpose(1,0)
        
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

    

    def get_param_stats(self): #" b: "+str(self.dense.bias.data) +\ "b: "+str(self.dense_mean.bias.data) +#"scaler: "+str(self.nn_scaler.data) +\
        return "TK: dense w: "+str(self.dense.weight.data) +\
        "self.kernel_mult:" +str(self.kernel_mult.data)+ " self.chunk_scoring: " +str(self.chunk_scoring.data) +\
        "self.tk_score:" +str(self.tk_score.data)+ " self.bert_score: " +str(self.bert_score.data) +\
        "mixer: "+str(self.mixer.data)

    def get_param_secondary(self):
        return {"dense_weight":self.dense.weight,#"dense_bias":self.dense.bias,
                #"dense_mean_weight":self.dense_mean.weight,#"dense_mean_bias":self.dense_mean.bias,
                #"dense_comb_weight":self.dense_comb.weight, 
                #"scaler":self.nn_scaler ,
                "mixer":self.mixer}


class TK_vNext_5(nn.Module):
    '''
    TK is a neural IR model - a fusion between transformer contextualization & kernel-based scoring

    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return TK_vNext_5(word_embeddings_out_dim, 
                     kernels_mu =    config["tk_kernels_mu"],
                     kernels_sigma = config["tk_kernels_sigma"],
                     att_heads =     config["tk_att_heads"],
                     att_layer =     config["tk_att_layer"],
                     att_proj_dim =  config["tk_att_proj_dim"],
                     att_ff_dim =    config["tk_att_ff_dim"],
                     max_length =    config["max_doc_length"],
                     use_pos_encoding     = config["tk_use_pos_encoding"],
                     use_diff_posencoding = config["tk_use_diff_posencoding"],
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
                 #position_bias_bin_percent,
                 #position_bias_absolute_steps
                 ):

        super(TK_vNext_5, self).__init__()

        n_kernels = len(kernels_mu)
        self.use_pos_encoding     = use_pos_encoding    
        self.use_diff_posencoding = use_diff_posencoding

        if len(kernels_mu) != len(kernels_sigma):
            raise Exception("len(kernels_mu) != len(kernels_sigma)")

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(kernels_mu), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(kernels_sigma), requires_grad=False).view(1, 1, 1, n_kernels)

        reduced_rep_dim = 200

        pos_f = self.get_positional_features(reduced_rep_dim,30) #max_timescale=100000
        pos_f.requires_grad = False
        self.positional_features_q = pos_f #nn.Parameter(pos_f)
        self.positional_features_q.requires_grad = False

        if self.use_diff_posencoding == True:
            pos_f = self.get_positional_features(reduced_rep_dim,max_length+500) #max_timescale=100000
            pos_f.requires_grad = False
            self.positional_features_d = pos_f[:,500:,:] #nn.Parameter(pos_f)
            self.positional_features_d.requires_grad = False
        else:
            self.positional_features_d = self.positional_features_q


        self.mixer = nn.Parameter(torch.full([1], 0.5, dtype=torch.float32, requires_grad=True))

        self.emb_reducer = nn.Linear(_embsize, reduced_rep_dim, bias=True)


        encoder_layer = nn.TransformerEncoderLayer(reduced_rep_dim, att_heads, dim_feedforward=att_ff_dim, dropout=0)
        self.contextualizer = nn.TransformerEncoder(encoder_layer, att_layer, norm=None)


        encoder_layer = nn.TransformerEncoderLayer(reduced_rep_dim, att_heads, dim_feedforward=att_ff_dim, dropout=0)
        self.cls_classifier = nn.TransformerEncoder(encoder_layer, 6, norm=None)


        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()


        #self.kernel_bias = nn.Parameter(torch.full([1,1,n_kernels], 0.5, dtype=torch.float32, requires_grad=True))
        self.kernel_mult = nn.Parameter(torch.full([1,1,n_kernels], 1, dtype=torch.float32, requires_grad=True))
        
        
        self.chunk_scoring_pos = nn.Parameter(torch.full([1,50], 1, dtype=torch.float32, requires_grad=True))
        self.chunk_scoring = nn.Parameter(torch.full([1,5], 1, dtype=torch.float32, requires_grad=True))


        self.dense = nn.Linear(n_kernels, 1, bias=False)
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014) # inits taken from matchzoo

        #self.query_learn_to_select = nn.Linear(100, 5, bias=False)

        self.interaction_cls = nn.Parameter(torch.empty((1,1,reduced_rep_dim)),requires_grad=True)
        torch.nn.init.uniform_(self.interaction_cls, -0.014, 0.014)
        self.interaction_mask_add = torch.full([1,1], 1, dtype=torch.float32, requires_grad=False).cuda()

        self.doc_cls = nn.Parameter(torch.empty((1,1,reduced_rep_dim)),requires_grad=True)
        torch.nn.init.uniform_(self.doc_cls, -0.014, 0.014)
        self.doc_cls_mask_add = torch.full([1,1], 1, dtype=torch.float32, requires_grad=False).cuda()

        self.parts_cls = nn.Parameter(torch.empty((1,1,reduced_rep_dim)),requires_grad=True)
        torch.nn.init.uniform_(self.parts_cls, -0.014, 0.014)
        self.parts_cls_mask_add = torch.full([1,2], 1, dtype=torch.float32, requires_grad=False).cuda()

        self.cls_type_doc = nn.Parameter(torch.empty((1,1,reduced_rep_dim)),requires_grad=True)
        torch.nn.init.uniform_(self.cls_type_doc, -0.014, 0.014)

        self.cls_type_query = nn.Parameter(torch.empty((1,1,reduced_rep_dim)),requires_grad=True)
        torch.nn.init.uniform_(self.cls_type_query, -0.014, 0.014)

        self.cls_doc_chunk_pos = nn.Parameter(torch.empty((1,100,reduced_rep_dim)),requires_grad=True)
        torch.nn.init.uniform_(self.cls_doc_chunk_pos, -0.014, 0.014)


        self.cls_interact_scorer = nn.Linear(reduced_rep_dim, reduced_rep_dim, bias=True)
        self.cls_interact_scorer2 = nn.Linear(reduced_rep_dim, 1, bias=True)

        self.combiner = nn.Parameter(torch.full([2], 1, dtype=torch.float32, requires_grad=True))


    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor, 
                output_secondary_output: bool = False) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # contextualization
        # -------------------------------------------------------

        query_embeddings = self.emb_reducer(query_embeddings)

        #query_embeddings=torch.cat([query_embeddings])
        query_embeddings_cls = torch.cat([self.interaction_cls.expand(query_embeddings.shape[0],1,-1),query_embeddings],dim=1)
        query_embeddings_mask_cls = torch.cat([self.interaction_mask_add.expand(query_pad_oov_mask.shape[0],-1),query_pad_oov_mask],dim=1)


        if self.use_pos_encoding:
            query_embeddings = query_embeddings + self.positional_features_q[:,:query_embeddings.shape[1],:]


        query_embeddings = self.forward_representation(query_embeddings_cls, query_embeddings_mask_cls,skip_first=True)

        query_classficication = query_embeddings[:,0] #self.query_learn_to_select(query_embeddings[:,0])

        query_embeddings = query_embeddings[:,1:]

        chunk_size = 40
        overlap = 5

        extended_chunk_size = chunk_size + 2 * overlap
        needed_padding = extended_chunk_size - ((document_pad_oov_mask.shape[1]-overlap) % chunk_size)
        orig_doc_len = document_pad_oov_mask.shape[1]
        #x = torch.range(1,106)
        #x = nn.functional.pad(x,(overlap,needed_padding))
        #unique_entries = x.unfold(0,extended_chunk_size,chunk_size)[:,overlap:-overlap]

        document_embeddings = self.emb_reducer(document_embeddings)

        document_embeddings = nn.functional.pad(document_embeddings,(0,0,overlap, needed_padding))
        document_pad_oov_mask = nn.functional.pad(document_pad_oov_mask,(overlap, needed_padding))

        chunked_docs = document_embeddings.unfold(1,extended_chunk_size,chunk_size).transpose(-1,-2)#[:,:,overlap:-overlap,:]
        chunked_pad = document_pad_oov_mask.unfold(1,extended_chunk_size,chunk_size)#[:,:,overlap:-overlap]
        
        batch_size = chunked_docs.shape[0]
        chunk_pieces = chunked_docs.shape[1]

        chunked_docs2=chunked_docs.reshape(-1,extended_chunk_size,document_embeddings.shape[-1])
        chunked_pad2=chunked_pad.reshape(-1,extended_chunk_size)
        #chunked_pad2[chunked_pad2.sum(-1) == 0] = 1

        packed_indices = chunked_pad2[:,overlap:-overlap].sum(-1) != 0

        documents_packed = chunked_docs2[packed_indices]
        padding_packed = chunked_pad2[packed_indices]

        if self.use_pos_encoding:
            documents_packed = documents_packed + self.positional_features_d[:,:documents_packed.shape[1],:]


        doc_packed_cls = torch.cat([self.doc_cls.expand(documents_packed.shape[0],1,-1),documents_packed],dim=1)
        doc_packed_mask_cls = torch.cat([self.doc_cls_mask_add.expand(padding_packed.shape[0],-1),padding_packed],dim=1)

        documents_packed = self.forward_representation(doc_packed_cls, doc_packed_mask_cls,skip_first=True)

        doc_cls_vectors = documents_packed[:,0]

        documents_unique_again = documents_packed[:,overlap+1:-overlap,:]
        #document_mask_unique_again = chunked_pad[:,:,overlap:-overlap]
        document_mask_packed_unique = padding_packed[:,overlap:-overlap]
        #        
        # reshape back in original form
 

        assert not torch.isnan(documents_packed).any()


        #packed_query_cls = query_classficication.unsqueeze(1).expand(-1,chunk_pieces,-1).reshape(-1,query_classficication.shape[-1])[packed_indices]
#
        #cls_interactions = packed_query_cls * doc_cls_vectors
#
        #cls_interactions = self.cls_interact_scorer(cls_interactions)
        #cls_scores = self.cls_interact_scorer2(cls_interactions)
#
        ## need to combine scores again
        #all_scores = torch.zeros((chunked_docs2.shape[0],1), dtype=chunked_docs2.dtype, layout=chunked_docs2.layout, device=chunked_docs2.device)
        #all_scores[packed_indices] = cls_scores
#
        ## score per doc & chunk !
        #cls_per_chunk_score = all_scores.view(batch_size,chunk_pieces,1).squeeze(-1)


        #packed_query_cls = query_classficication.unsqueeze(1).expand(-1,chunk_pieces,-1).reshape(-1,query_classficication.shape[-1])[packed_indices]

        #cls_interactions = packed_query_cls * doc_cls_vectors

        # need to combine scores again
        doc_cls_unpacked = torch.zeros((chunked_docs2.shape[0],chunked_docs2.shape[-1]), dtype=chunked_docs2.dtype, layout=chunked_docs2.layout, device=chunked_docs2.device)
        doc_cls_unpacked[packed_indices] = doc_cls_vectors

        # score per doc & chunk !
        cls_per_doc = doc_cls_unpacked.view(batch_size,chunk_pieces,chunked_docs2.shape[-1]).squeeze(-1)

        cls_input = torch.cat([self.parts_cls.expand(batch_size,-1,-1),query_classficication.unsqueeze(1)+self.cls_type_query,cls_per_doc+self.cls_type_doc + self.cls_doc_chunk_pos[:,cls_per_doc.shape[1],:]],dim=1)

        cls_doc_mask = packed_indices.view(-1,chunk_pieces).float()

        cls_mask = torch.cat([self.parts_cls_mask_add.expand(batch_size,-1),cls_doc_mask],dim=1)

        cls_interactions = self.cls_classifier((cls_input).transpose(1,0),src_key_padding_mask=~cls_mask.bool()).transpose(1,0)
        cls_interactions = self.cls_interact_scorer(cls_interactions[:,0])
        cls_scores = self.cls_interact_scorer2(cls_interactions)




        #
        # masks 
        # -------------------------------------------------------

        #query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        #query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)
        #doc_lengths = torch.sum(document_pad_oov_mask, 1)

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


        #
        # kernel-pooling
        # -------------------------------------------------------

        per_kernel_query = torch.sum(kernel_results_masked, 2) 

        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query * self.kernel_mult , min=1e-10)) #self.kernel_bias
        log_per_kernel_query_masked = log_per_kernel_query * packed_query_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 


        ##
        ## weight kernel bins
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)

        # need to combine scores again
        all_scores = torch.zeros((chunked_docs2.shape[0],1), dtype=chunked_docs2.dtype, layout=chunked_docs2.layout, device=chunked_docs2.device)
        all_scores[packed_indices] = dense_out

        # score per doc & chunk !
        score = all_scores.view(batch_size,chunk_pieces,1).squeeze(-1)

        #score = score * cls_per_chunk_score

        #self.chunk_scoring
        top_k = 5

        if score.shape[1] < top_k:
            score = nn.functional.pad(score,(0, top_k - score.shape[1]))

        #score = score * self.chunk_scoring_pos[:,:score.shape[1]]

        score[score == 0] = -1000
        score = score.topk(k=top_k,dim=1)[0]
        score[score == -1000] = 0
        
        #if score.shape[1] > query_classficication.shape[1]:
         #   score = score[:,:query_classficication.shape[1]]

        #qc = query_classficication[:,:score.shape[1]]

        #qc[score==0] = -1000

        #qc = nn.functional.softmax(qc,dim=1)

        score = (score * self.chunk_scoring).sum(dim=1) #self.chunk_scoring

        #score = score * self.combiner[0] + cls_scores.squeeze(-1) * self.combiner[1]
        score =  cls_scores.squeeze(-1) #* self.combiner[1]

        #score = (score * self.masked_softmax(score, score != 0)).sum(dim=1)
        #score = score.max(dim=1)[0]

        if output_secondary_output:
            query_mean_vector = query_embeddings.sum(dim=1) / query_pad_oov_mask.sum(dim=1).unsqueeze(-1)
            return score, {"score":score,"query_classficication":query_classficication,"dense_out":dense_out,"per_kernel":per_kernel, "total_chunks":chunked_docs2.shape[0],"packed_chunks":dense_out.shape[0],
                           "query_mean_vector":query_mean_vector,"cosine_matrix_masked":cosine_matrix}
        else:
            return score

    def forward_representation(self, sequence_embeddings: torch.Tensor, sequence_mask: torch.Tensor,skip_first=False,positional_features=None) -> torch.Tensor:

        #if positional_features is None:
        #    positional_features = self.positional_features_d[:,:sequence_embeddings.shape[1],:]

        sequence_embeddings = sequence_embeddings * sequence_mask.unsqueeze(-1)

        #pos_sequence = sequence_embeddings
        #if self.use_pos_encoding:
        #    pos_sequence = sequence_embeddings + positional_features
        
        sequence_embeddings_context = self.contextualizer((sequence_embeddings).transpose(1,0),src_key_padding_mask=~sequence_mask.bool()).transpose(1,0)

        sequence_embeddings = (self.mixer * sequence_embeddings + (1 - self.mixer) * sequence_embeddings_context) #* sequence_mask.unsqueeze(-1)

        if skip_first:
            sequence_embeddings[:,0] = sequence_embeddings_context[:,0]
        #else:

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

    def get_param_stats(self): #" b: "+str(self.dense.bias.data) +\ "b: "+str(self.dense_mean.bias.data) +#"scaler: "+str(self.nn_scaler.data) +\
        return "TK: dense w: "+str(self.dense.weight.data) +\
        "self.kernel_mult:" +str(self.kernel_mult.data)+\
        "self.combiner:" +str(self.combiner.data)+\
        "self.chunk_scoring:" +str(self.chunk_scoring.data)+\
        "mixer: "+str(self.mixer.data)

    def get_param_secondary(self):
        return {"dense_weight":self.dense.weight,#"dense_bias":self.dense.bias,
                #"dense_mean_weight":self.dense_mean.weight,#"dense_mean_bias":self.dense_mean.bias,
                #"dense_comb_weight":self.dense_comb.weight, 
                #"scaler":self.nn_scaler ,
                "mixer":self.mixer}



class TK_vNext_6(nn.Module):
    '''
    TK is a neural IR model - a fusion between transformer contextualization & kernel-based scoring

    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return TK_vNext_6(word_embeddings_out_dim, 
                     kernels_mu =    config["tk_kernels_mu"],
                     kernels_sigma = config["tk_kernels_sigma"],
                     att_heads =     config["tk_att_heads"],
                     att_layer =     config["tk_att_layer"],
                     att_proj_dim =  config["tk_att_proj_dim"],
                     att_ff_dim =    config["tk_att_ff_dim"],
                     max_length =    config["max_doc_length"],
                     use_pos_encoding     = config["tk_use_pos_encoding"],
                     use_diff_posencoding = config["tk_use_diff_posencoding"],
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
                 #position_bias_bin_percent,
                 #position_bias_absolute_steps
                 ):

        super(TK_vNext_6, self).__init__()

        n_kernels = len(kernels_mu)
        self.use_pos_encoding     = use_pos_encoding    
        self.use_diff_posencoding = use_diff_posencoding

        if len(kernels_mu) != len(kernels_sigma):
            raise Exception("len(kernels_mu) != len(kernels_sigma)")

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(kernels_mu), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(kernels_sigma), requires_grad=False).view(1, 1, 1, n_kernels)


        pos_f = self.get_positional_features(100,max_length) #max_timescale=100000
        pos_f.requires_grad = False
        self.positional_features_q = pos_f #nn.Parameter(pos_f)
        self.positional_features_q.requires_grad = False

        if self.use_diff_posencoding == True:
            pos_f = self.get_positional_features(100,max_length+500) #max_timescale=100000
            pos_f.requires_grad = False
            self.positional_features_d = pos_f[:,500:,:] #nn.Parameter(pos_f)
            self.positional_features_d.requires_grad = False
        else:
            self.positional_features_d = self.positional_features_q


        self.mixer = nn.Parameter(torch.full([1], 0.5, dtype=torch.float32, requires_grad=True))

        self.emb_reducer = nn.Linear(_embsize, 100, bias=True)


        encoder_layer = nn.TransformerEncoderLayer(100, att_heads, dim_feedforward=att_ff_dim, dropout=0)
        self.contextualizer = nn.TransformerEncoder(encoder_layer, att_layer, norm=None)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()


        self.kernel_bias = nn.Parameter(torch.full([1,1,n_kernels], 0.5, dtype=torch.float32, requires_grad=True))
        self.kernel_mult = nn.Parameter(torch.full([1,1,n_kernels], 1, dtype=torch.float32, requires_grad=True))
        
        
        self.chunk_scoring = nn.Parameter(torch.full([1,5], 1, dtype=torch.float32, requires_grad=True))


        self.dense = nn.Linear(n_kernels, 1, bias=False)
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014) # inits taken from matchzoo

        self.masked_softmax = MaskedSoftmax()

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor,
                query_idfs: torch.Tensor, document_idfs: torch.Tensor,
                output_secondary_output: bool = False) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # contextualization
        # -------------------------------------------------------

        query_embeddings = self.emb_reducer(query_embeddings)
        if self.use_pos_encoding:
            query_embeddings = query_embeddings + self.positional_features_q[:,:query_embeddings.shape[1],:]

        query_embeddings = self.forward_representation(query_embeddings, query_pad_oov_mask)

        chunk_size = 40
        overlap = 5

        extended_chunk_size = chunk_size + 2 * overlap
        needed_padding = extended_chunk_size - ((document_pad_oov_mask.shape[1]-overlap) % chunk_size)
        orig_doc_len = document_pad_oov_mask.shape[1]
        #x = torch.range(1,106)
        #x = nn.functional.pad(x,(overlap,needed_padding))
        #unique_entries = x.unfold(0,extended_chunk_size,chunk_size)[:,overlap:-overlap]

        document_embeddings = self.emb_reducer(document_embeddings)
        #if self.use_pos_encoding:
        #    document_embeddings = document_embeddings+ self.positional_features_d[:,:document_embeddings.shape[1],:]

        document_embeddings = nn.functional.pad(document_embeddings,(0,0,overlap, needed_padding))
        orig_document_pad_oov_mask = document_pad_oov_mask
        document_pad_oov_mask = nn.functional.pad(document_pad_oov_mask,(overlap, needed_padding))

        chunked_docs = document_embeddings.unfold(1,extended_chunk_size,chunk_size).transpose(-1,-2)#[:,:,overlap:-overlap,:]
        chunked_pad = document_pad_oov_mask.unfold(1,extended_chunk_size,chunk_size)#[:,:,overlap:-overlap]
        
        batch_size = chunked_docs.shape[0]
        chunk_pieces = chunked_docs.shape[1]

        chunked_docs2=chunked_docs.reshape(-1,extended_chunk_size,document_embeddings.shape[-1])
        chunked_pad2=chunked_pad.reshape(-1,extended_chunk_size)
        #chunked_pad2[chunked_pad2.sum(-1) == 0] = 1

        packed_indices = chunked_pad2[:,overlap:-overlap].sum(-1) != 0

        documents_packed = chunked_docs2[packed_indices]
        padding_packed = chunked_pad2[packed_indices]

        if self.use_pos_encoding:
            documents_packed = documents_packed + self.positional_features_d[:,:documents_packed.shape[1],:]


        documents_packed = self.forward_representation(documents_packed, padding_packed)

        documents_unique_again = documents_packed[:,overlap:-overlap,:]
        #document_mask_unique_again = chunked_pad[:,:,overlap:-overlap]
        document_mask_packed_unique = padding_packed[:,overlap:-overlap]
        assert not torch.isnan(documents_packed).any()

        doc_merge = torch.zeros((chunked_docs2.shape[0],documents_unique_again.shape[1],documents_unique_again.shape[2]),device=documents_unique_again.device)
        doc_merge[packed_indices] = documents_unique_again
        document_embeddings = doc_merge.view(batch_size,-1,chunk_size,document_embeddings.shape[-1]).view(batch_size,-1,document_embeddings.shape[-1])

        mask_merge = torch.zeros((chunked_docs2.shape[0],documents_unique_again.shape[1]),device=documents_unique_again.device)
        mask_merge[packed_indices] = document_mask_packed_unique
        document_pad_oov_mask = mask_merge.view(batch_size,-1,chunk_size).view(batch_size,-1)

        rand_size = 40
        rand_chunks = 5

        rand_results = [] # torch.zeros((document_embeddings.shape[0],rand_chunks,self.mu.shape[-1]))

        for rc in range(rand_chunks):

            random_doc_idxs = torch.multinomial(document_idfs.squeeze(-1)[:,:document_embeddings.shape[1]]+0.001,rand_size,replacement=False)

            random_indices_flat = (random_doc_idxs + torch.arange(0,document_embeddings.shape[0]*document_embeddings.shape[1],document_embeddings.shape[1],device=document_embeddings.device).unsqueeze(-1)).view(-1)

            random_selection = (document_embeddings.view(-1,document_embeddings.shape[-1])).index_select(0,random_indices_flat)
            random_selection_mask = (document_pad_oov_mask.view(-1)).index_select(0,random_indices_flat)


            random_docs = random_selection.view(batch_size,-1,document_embeddings.shape[-1])
            random_mask = random_selection_mask.view(batch_size,-1)


            #
            # masks 
            # -------------------------------------------------------

            #query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
            #query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)
            #doc_lengths = torch.sum(document_pad_oov_mask, 1)

            #
            # cosine matrix
            # -------------------------------------------------------
            #packed_query_embeddings = query_embeddings.unsqueeze(1).expand(-1,chunk_pieces,-1,-1).reshape(-1,query_embeddings.shape[1],query_embeddings.shape[-1])[packed_indices]
            #packed_query_mask = query_pad_oov_mask.unsqueeze(1).expand(-1,chunk_pieces,-1).reshape(-1,query_embeddings.shape[1])[packed_indices]

            # shape: (batch, query_max, doc_max)
            cosine_matrix = self.cosine_module.forward(query_embeddings, random_docs)
            #cosine_matrix_masked = cosine_matrix * query_by_doc_mask

            #
            # gaussian kernels & soft-TF
            #
            # first run through kernel, then sum on doc dim then sum on query dim
            # -------------------------------------------------------

            cosine_matrix_extradim = cosine_matrix.unsqueeze(-1)        
            raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
            kernel_results_masked = raw_kernel_results * random_mask.unsqueeze(1).unsqueeze(-1)


            #
            # kernel-pooling
            # -------------------------------------------------------

            per_kernel_query = torch.sum(kernel_results_masked, 2) 

            log_per_kernel_query = torch.log(torch.clamp(per_kernel_query * self.kernel_mult , min=1e-10)) #self.kernel_bias
            log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
            per_kernel = torch.sum(log_per_kernel_query_masked, 1) 


            ##
            ## weight kernel bins
            ## -------------------------------------------------------

            rand_results.append(self.dense(per_kernel))

        rand_results = torch.cat(rand_results,dim=1)

        #score = dense_out.squeeze(-1)

        ## need to combine scores again
        #all_scores = torch.zeros((chunked_docs2.shape[0],1), dtype=chunked_docs2.dtype, layout=chunked_docs2.layout, device=chunked_docs2.device)
        #all_scores[packed_indices] = dense_out
#
        ## score per doc & chunk !
        #score = all_scores.view(batch_size,chunk_pieces,1).squeeze(-1)
#
        ##self.chunk_scoring
        #top_k = 3
#
        #if score.shape[1] < top_k:
        #    score = nn.functional.pad(score,(0, top_k - score.shape[1]))
#
        #score[score == 0] = -1000
        score = rand_results.sort(dim=1)[0]
        #score[score == -1000] = 0
        #
        score = (score * self.chunk_scoring).sum(dim=1)

        #score = (score * self.masked_softmax(score, score != 0)).sum(dim=1)
        #score = score.max(dim=1)[0]

        if output_secondary_output:
            query_mean_vector = query_embeddings.sum(dim=1) / query_pad_oov_mask.sum(dim=1).unsqueeze(-1)
            return score, {"score":score,"rand_results":rand_results,"per_kernel":per_kernel, "total_chunks":chunked_docs2.shape[0],"packed_chunks":score.shape[0],
                           "query_mean_vector":query_mean_vector,"cosine_matrix_masked":cosine_matrix}
        else:
            return score

    def forward_representation(self, sequence_embeddings: torch.Tensor, sequence_mask: torch.Tensor,positional_features=None) -> torch.Tensor:

        #if positional_features is None:
        #    positional_features = self.positional_features_d[:,:sequence_embeddings.shape[1],:]

        sequence_embeddings = sequence_embeddings * sequence_mask.unsqueeze(-1)

        #pos_sequence = sequence_embeddings
        #if self.use_pos_encoding:
        #    pos_sequence = sequence_embeddings + positional_features
        
        sequence_embeddings_context = self.contextualizer((sequence_embeddings).transpose(1,0),src_key_padding_mask=~sequence_mask.bool()).transpose(1,0)
        
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

    

    def get_param_stats(self): #" b: "+str(self.dense.bias.data) +\ "b: "+str(self.dense_mean.bias.data) +#"scaler: "+str(self.nn_scaler.data) +\
        return "TK: dense w: "+str(self.dense.weight.data) +\
        "self.kernel_mult:" +str(self.kernel_mult.data)+ " self.chunk_scoring: " +str(self.chunk_scoring.data) +\
        "mixer: "+str(self.mixer.data)

    def get_param_secondary(self):
        return {"dense_weight":self.dense.weight,#"dense_bias":self.dense.bias,
                #"dense_mean_weight":self.dense_mean.weight,#"dense_mean_bias":self.dense_mean.bias,
                #"dense_comb_weight":self.dense_comb.weight, 
                #"scaler":self.nn_scaler ,
                "mixer":self.mixer}
