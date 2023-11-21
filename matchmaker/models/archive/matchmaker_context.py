from typing import Dict, Iterator, List
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          
from allennlp.modules.matrix_attention.dot_product_matrix_attention import *                          
from allennlp.nn.util import get_range_vector,get_device_of

class Matchmaker_context_v1(nn.Module):
    '''
    Paper: ...

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Matchmaker_context_v1(word_embeddings_out_dim, n_kernels = config["mm_light_kernels"])

    def __init__(self,
                 _embsize:int,
                 n_kernels: int):

        super(Matchmaker_context_v1, self).__init__()

        #
        # static - kernels & magnitude variables
        #
        self.mu = nn.Parameter(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1,1, 1, n_kernels)
        self.sigma = nn.Parameter(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1,1, 1, n_kernels)

        #
        # contextualisation
        #
        self.stacked_att = StackedSelfAttentionEncoder(input_dim=_embsize,
                 hidden_dim=_embsize,
                 projection_dim=32,
                 feedforward_hidden_dim=100,
                 num_layers=1,
                 num_attention_heads=32,
                 dropout_prob = 0,
                 residual_dropout_prob = 0,
                 attention_dropout_prob= 0)

        #
        # matchmatrix module
        #
        self.match_matrix = CosineMatrixAttention() # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 

        #
        # conv modules (n-gram matching)
        #
        n_grams = 4
        conv_out_dim = 300
        self.ngram_representation = []
        for i in range(2, n_grams + 1):
            self.ngram_representation.append(
                nn.Sequential(
                    nn.ConstantPad1d((0,i - 1), 0),
                    nn.Conv1d(kernel_size=i, in_channels=_embsize * 2, out_channels=conv_out_dim),
                    nn.ReLU()) 
            )
        self.ngram_representation = nn.ModuleList(self.ngram_representation) # register conv as part of the model

        #
        # reduction / scoring mlp
        #

        combine_count = n_kernels + (n_kernels*len(self.ngram_representation))

        self.linear_sum = nn.Linear(combine_count, 1, bias=True) 
        self.linear_mean = nn.Linear(combine_count, 1, bias=True)
        self.linear_comb = nn.Linear(2, 1, bias=False)


    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor) -> torch.Tensor:

        #
        # (0) contextualize embeddings
        # -----------------------------
        #
        query_embeddings_context = self.stacked_att(query_embeddings,query_pad_oov_mask)
        document_embeddings_context = self.stacked_att(document_embeddings,document_pad_oov_mask)

        query_embeddings = torch.cat([query_embeddings,query_embeddings_context],dim=2)
        document_embeddings = torch.cat([document_embeddings,document_embeddings_context],dim=2)

        #
        # (1) word-to-word matching
        # --------------------------
        #
        one_to_one_match_matrix = torch.tanh(self.match_matrix(query_embeddings, document_embeddings).unsqueeze(-1)) #torch.tanh()
        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2)).unsqueeze(-1)

        #
        # kernels
        oto_kernel_results = torch.exp(- torch.pow(one_to_one_match_matrix - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        oto_kernel_results_masked = oto_kernel_results * query_by_doc_mask
        oto_kernel_query = torch.sum(oto_kernel_results_masked, 2)

        #
        # sum kernels
        log_per_kernel_query = torch.log(torch.clamp(oto_kernel_query, min=1e-10))
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        oto_sum_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        #
        # mean kernels
        doc_lengths = torch.sum(document_pad_oov_mask, dim=1).view(-1,1,1)
        per_kernel_query_mean = oto_kernel_query / doc_lengths

        log_per_kernel_query_mean = torch.log(torch.clamp(per_kernel_query_mean, min=1e-10))
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        oto_mean_kernel = torch.sum(log_per_kernel_query_masked_mean, 1) 

        #
        # (2) n-gram matching
        # --------------------
        #
        query_embeddings_t = query_embeddings.transpose(1, 2)
        document_embeddings_t = document_embeddings.transpose(1, 2)

        ngram_sum_kernel = []
        ngram_mean_kernel = []

        for i,ngram_rep in enumerate(self.ngram_representation):
            query_ngram = ngram_rep(query_embeddings_t).transpose(1, 2) 
            document_ngram = ngram_rep(document_embeddings_t).transpose(1, 2)

            n_gram_match_matrix = torch.tanh(self.match_matrix(query_ngram, document_ngram).unsqueeze(-1)) #torch.tanh()

            #
            # kernels
            ngram_kernel_results = torch.exp(- torch.pow(n_gram_match_matrix - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
            ngram_kernel_results_masked = ngram_kernel_results * query_by_doc_mask
            ngram_kernel_query = torch.sum(ngram_kernel_results_masked, 2)

            #
            # sum kernels
            log_per_kernel_query = torch.log(torch.clamp(ngram_kernel_query, min=1e-10))
            log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
            ngram_sum_kernel.append(torch.sum(log_per_kernel_query_masked, 1))

            #
            # mean kernels
            per_kernel_query_mean = ngram_kernel_query / doc_lengths
            log_per_kernel_query_mean = torch.log(torch.clamp(per_kernel_query_mean, min=1e-10))
            log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
            ngram_mean_kernel.append(torch.sum(log_per_kernel_query_masked_mean, 1))

        #
        # (3) reduction / scoring layer
        # ------------------------------
        # 
        linear_sum_out = self.linear_sum(torch.cat([oto_sum_kernel] + ngram_sum_kernel,dim=1))
        linear_mean_out = self.linear_mean(torch.cat([oto_mean_kernel] + ngram_mean_kernel,dim=1))
        linear_comb_out = self.linear_comb(torch.cat([linear_sum_out,linear_mean_out],dim=1))

        score = torch.squeeze(linear_comb_out,1)
        return score

    def get_param_stats(self):
        return "MM_context_v1: dense w: "+str(self.linear_sum.weight.data)+" b: "+str(self.linear_sum.bias.data) +\
        "dense_mean weight: "+str(self.linear_mean.weight.data)+"b: "+str(self.linear_mean.bias.data) +\
        "dense_comb weight: "+str(self.linear_comb.weight.data)

    def kernel_mus(self, n_kernels: int):
        """
        get the mu for each guassian kernel. Mu is the middle of each bin
        :param n_kernels: number of kernels (including exact match). first one is exact match
        :return: l_mu, a list of mu.
        """
        l_mu = [1.0]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

    def kernel_sigmas(self, n_kernels: int):
        """
        get sigmas for each guassian kernel.
        :param n_kernels: number of kernels (including exactmath.)
        :param lamb:
        :param use_exact:
        :return: l_sigma, a list of simga
        """
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.0001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma



class Matchmaker_context_v2(nn.Module):
    '''
    Paper: ...

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Matchmaker_context_v2(word_embeddings_out_dim, n_kernels = config["mm_light_kernels"])

    def __init__(self,
                 _embsize:int,
                 n_kernels: int):

        super(Matchmaker_context_v2, self).__init__()

        #
        # static - kernels & magnitude variables
        #
        self.mu = nn.Parameter(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1,1, 1, n_kernels)
        self.sigma = nn.Parameter(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1,1, 1, n_kernels)

        #
        # contextualisation
        #
        self.stacked_att = StackedSelfAttentionEncoder(input_dim=_embsize,
                 hidden_dim=_embsize,
                 projection_dim=32,
                 feedforward_hidden_dim=100,
                 num_layers=1,
                 num_attention_heads=32,
                 dropout_prob = 0,
                 residual_dropout_prob = 0,
                 attention_dropout_prob= 0)
                 #use_positional_encoding=False)

        #
        # matchmatrix module
        #
        self.match_matrix = CosineMatrixAttention() # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 

        #
        # proximity matching
        #
        self.prox_range = [2,3,4,5,6,7,8,9,10,11,12,13]
        self.prox_pools = []
        self.prox_scaler = nn.Parameter(torch.FloatTensor(self.prox_range))
        self.prox_linear = []
        for i in self.prox_range:
            self.prox_pools.append(nn.AvgPool1d(kernel_size=i,stride=1)) # ,padding=int(math.ceil(i/2))
            self.prox_linear.append(nn.Linear(n_kernels, 1, bias=True))
        
        self.prox_pools = nn.ModuleList(self.prox_pools)
        self.prox_linear = nn.ModuleList(self.prox_linear)
        self.prox_combine = nn.Linear(len(self.prox_linear), 1, bias=True)

        #
        # conv modules (n-gram matching)
        #
        n_grams = 4
        conv_out_dim = 300
        self.ngram_representation = []
        for i in range(2, n_grams + 1):
            self.ngram_representation.append(
                nn.Sequential(
                    nn.ConstantPad1d((0,i - 1), 0),
                    nn.Conv1d(kernel_size=i, in_channels=_embsize, out_channels=conv_out_dim),
                    nn.ReLU()) 
            )
        self.ngram_representation = nn.ModuleList(self.ngram_representation) # register conv as part of the model

        #
        # reduction / scoring mlp
        #

        combine_count = (n_kernels*len(self.ngram_representation))

        self.linear_sum = nn.Linear(combine_count, 1, bias=True) 
        self.linear_mean = nn.Linear(combine_count, 1, bias=True)

        self.linear_oto_sum = nn.Linear(n_kernels, 1, bias=True) 
        self.linear_oto_mean = nn.Linear(n_kernels, 1, bias=True)

        self.linear_comb = nn.Linear(5, 1, bias=False)


    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor) -> torch.Tensor:

        #
        # (0) contextualize embeddings
        # -----------------------------
        #
        query_embeddings_context = self.stacked_att(query_embeddings,query_pad_oov_mask) #add_positional_features
        document_embeddings_context = self.stacked_att(document_embeddings,document_pad_oov_mask)

        query_embeddings_context = torch.cat([query_embeddings,query_embeddings_context],dim=2)
        document_embeddings_context = torch.cat([document_embeddings,document_embeddings_context],dim=2)

        #
        # (1) word-to-word matching
        # --------------------------
        #
        one_to_one_match_matrix = self.match_matrix(query_embeddings_context, document_embeddings_context).unsqueeze(-1) #torch.tanh()
        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2)).unsqueeze(-1)

        #
        # kernels
        oto_kernel_results = torch.exp(- torch.pow(one_to_one_match_matrix - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        oto_kernel_results_masked = oto_kernel_results * query_by_doc_mask
        oto_kernel_query = torch.sum(oto_kernel_results_masked, 2)

        #
        # sum kernels
        log_per_kernel_query = torch.log(torch.clamp(oto_kernel_query, min=1e-10))
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        oto_sum_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        #
        # mean kernels
        doc_lengths = torch.sum(document_pad_oov_mask, dim=1).view(-1,1,1)
        per_kernel_query_mean = oto_kernel_query / doc_lengths

        log_per_kernel_query_mean = torch.log(torch.clamp(per_kernel_query_mean, min=1e-10))
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        oto_mean_kernel = torch.sum(log_per_kernel_query_masked_mean, 1) 

        #
        # (2) n-gram matching
        # --------------------
        #
        query_embeddings_t = query_embeddings.transpose(1, 2)
        document_embeddings_t = document_embeddings.transpose(1, 2)

        ngram_sum_kernel = []
        ngram_mean_kernel = []

        for i,ngram_rep in enumerate(self.ngram_representation):
            query_ngram = ngram_rep(query_embeddings_t).transpose(1, 2) 
            document_ngram = ngram_rep(document_embeddings_t).transpose(1, 2)

            n_gram_match_matrix = self.match_matrix(query_ngram, document_ngram).unsqueeze(-1) #torch.tanh()

            #
            # kernels
            ngram_kernel_results = torch.exp(- torch.pow(n_gram_match_matrix - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
            ngram_kernel_results_masked = ngram_kernel_results * query_by_doc_mask
            ngram_kernel_query = torch.sum(ngram_kernel_results_masked, 2)

            #
            # sum kernels
            log_per_kernel_query = torch.log(torch.clamp(ngram_kernel_query, min=1e-10))
            log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
            ngram_sum_kernel.append(torch.sum(log_per_kernel_query_masked, 1))

            #
            # mean kernels
            per_kernel_query_mean = ngram_kernel_query / doc_lengths
            log_per_kernel_query_mean = torch.log(torch.clamp(per_kernel_query_mean, min=1e-10))
            log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
            ngram_mean_kernel.append(torch.sum(log_per_kernel_query_masked_mean, 1))

        #
        # (3) proximity matching
        #
        max_per_doc = torch.max(one_to_one_match_matrix, dim=1)[0].squeeze(-1).unsqueeze(1)

        prox_res = []
        prox_res_tensor = torch.empty((one_to_one_match_matrix.shape[0],len(self.prox_range),one_to_one_match_matrix.shape[2],1),device=one_to_one_match_matrix.device)
        for i, avg_pool in enumerate(self.prox_pools):
            avg_res = avg_pool(max_per_doc) * self.prox_scaler[i]
            avg_res_padded = torch.nn.functional.pad(avg_res,(0,self.prox_range[i]-1))
            prox_res_tensor[:,i] = avg_res_padded.squeeze().unsqueeze(-1)

        prox_kernel = torch.exp(- torch.pow(prox_res_tensor - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        prox_kernel_masked = prox_kernel.squeeze() * document_pad_oov_mask.view(document_pad_oov_mask.shape[0],1,-1,1)
        prox_kernel_summed = torch.sum(prox_kernel_masked, dim=2)

        prox_linear_result = torch.empty((one_to_one_match_matrix.shape[0],len(self.prox_linear)),device=one_to_one_match_matrix.device)
        for i in range(len(self.prox_linear)):
            prox_linear_result[:,i] = self.prox_linear[i](prox_kernel_summed[:,i]).squeeze(-1)
        prox_linear_result = self.prox_combine(prox_linear_result)

        #
        # (4) reduction / scoring layer
        # ------------------------------
        # 
        linear_oto_sum_out = self.linear_oto_sum(oto_sum_kernel)
        linear_oto_mean_out = self.linear_oto_mean(oto_mean_kernel)

        linear_sum_out = self.linear_sum(torch.cat(ngram_sum_kernel,dim=1))
        linear_mean_out = self.linear_mean(torch.cat(ngram_mean_kernel,dim=1))
        linear_comb_out = self.linear_comb(torch.cat([linear_oto_sum_out,linear_oto_mean_out,linear_sum_out,linear_mean_out,prox_linear_result],dim=1))

        score = torch.squeeze(linear_comb_out,1)
        return score

    def get_param_stats(self):
        return "MM_context_v2: linear_sum w: "+str(self.linear_sum.weight.data)+" b: "+str(self.linear_sum.bias.data) +\
        " dense_mean weight: "+str(self.linear_mean.weight.data)+"b: "+str(self.linear_mean.bias.data) +\
        " linear_oto_sum weight: "+str(self.linear_oto_sum.weight.data)+"b: "+str(self.linear_oto_sum.bias.data) +\
        " linear_oto_mean weight: "+str(self.linear_oto_mean.weight.data)+"b: "+str(self.linear_oto_mean.bias.data) +\
        " prox_combine weight: "+str(self.prox_combine.weight.data)+"b: "+str(self.prox_combine.bias.data) +\
        " linear_comb weight: "+str(self.linear_comb.weight.data)

    def kernel_mus(self, n_kernels: int):
        """
        get the mu for each guassian kernel. Mu is the middle of each bin
        :param n_kernels: number of kernels (including exact match). first one is exact match
        :return: l_mu, a list of mu.
        """
        l_mu = [1.0]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

    def kernel_sigmas(self, n_kernels: int):
        """
        get sigmas for each guassian kernel.
        :param n_kernels: number of kernels (including exactmath.)
        :param lamb:
        :param use_exact:
        :return: l_sigma, a list of simga
        """
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.0001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma





def add_positional_features(tensor: torch.Tensor,
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
    _, timesteps, hidden_dim = tensor.size()

    timestep_range = get_range_vector(timesteps, get_device_of(tensor)).data.to(dtype=tensor.dtype)
    # We're generating both cos and sin frequencies,
    # so half for each.
    num_timescales = hidden_dim // 2
    timescale_range = get_range_vector(num_timescales, get_device_of(tensor)).data.to(dtype=tensor.dtype)

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
    return tensor + sinusoids.unsqueeze(0)