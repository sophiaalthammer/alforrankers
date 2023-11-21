from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          
from allennlp.modules.matrix_attention.dot_product_matrix_attention import *                          

class Matchmaker_v1c(nn.Module):
    '''
    doc length avg. kernel from mm_light_vc1 applied to conv_knrm 

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Matchmaker_v1c(word_embeddings_out_dim=word_embeddings_out_dim,
                         n_grams=config["conv_knrm_ngrams"], 
                         n_kernels=config["conv_knrm_kernels"],
                         conv_out_dim=config["conv_knrm_conv_out_dim"])

    def __init__(self,
                 word_embeddings_out_dim: int,
                 n_grams:int,
                 n_kernels: int,
                 conv_out_dim:int):

        super(Matchmaker_v1c, self).__init__()

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))

        self.convolutions = []
        for i in range(1, n_grams + 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.ConstantPad1d((0,i - 1), 0),
                    nn.Conv1d(kernel_size=i, in_channels=word_embeddings_out_dim, out_channels=conv_out_dim),
                    nn.ReLU()) 
            )
        self.convolutions = nn.ModuleList(self.convolutions) # register conv as part of the model

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        # *9 because we concat the 3x3 conv match sums together before the dense layer
        self.dense = nn.Linear(n_kernels * n_grams * n_grams, 1, bias=True) 
        self.dense_mean = nn.Linear(n_kernels * n_grams * n_grams, 1, bias=True)
        self.dense_comb = nn.Linear(2, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high fot
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo

    def get_param_stats(self):
        return "MM: dense w: "+str(self.dense.weight.data)+" b: "+str(self.dense.bias.data) +\
        "dense_mean weight: "+str(self.dense_mean.weight.data)+"b: "+str(self.dense_mean.bias.data) +\
        "dense_comb weight: "+str(self.dense_comb.weight.data) + "scaler: "+str(self.nn_scaler.data)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor) -> torch.Tensor:

        #
        # prepare embedding tensors
        # -------------------------------------------------------

        # we assume 1 is the unknown token, 0 is padding - both need to be removed
        #if len(query["tokens"].shape) == 2: # (embedding lookup matrix)
#
        #    # shape: (batch, query_max)
        #    query_pad_oov_mask = (query["tokens"] > 1).float()
        #    # shape: (batch, doc_max)
        #    document_pad_oov_mask = (document["tokens"] > 1).float()
#
        #    # shape: (batch, query_max)
        #    query_pad_mask = (query["tokens"] > 0).float()
        #    # shape: (batch, doc_max)
        #    document_pad_mask = (document["tokens"] > 0).float()
#
        #else: # == 3 (elmo characters per word)
        #    
        #    # shape: (batch, query_max)
        #    query_pad_oov_mask = (torch.sum(query["tokens"],2) > 0).float()
        #    # shape: (batch, doc_max)
        #    document_pad_oov_mask = (torch.sum(document["tokens"],2) > 0).float()
#
        ##query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)
#
        ## shape: (batch, query_max,emb_dim)
        #query_embeddings = self.word_embeddings(query) * query_pad_oov_mask.unsqueeze(-1)
        ## shape: (batch, document_max,emb_dim)
        #document_embeddings = self.word_embeddings(document) * document_pad_oov_mask.unsqueeze(-1)

        query_pad_mask = query_pad_oov_mask
        document_pad_mask = document_pad_oov_mask

        query_by_doc_mask = torch.bmm(query_pad_mask.unsqueeze(-1), document_pad_mask.unsqueeze(-1).transpose(-1, -2))

        # !! conv1d requires tensor in shape: [batch, emb_dim, sequence_length ]
        # so we transpose embedding tensors from : [batch, sequence_length,emb_dim] to [batch, emb_dim, sequence_length ]
        # feed that into the conv1d and reshape output from [batch, conv1d_out_channels, sequence_length ] 
        # to [batch, sequence_length, conv1d_out_channels]

        query_embeddings_t = query_embeddings.transpose(1, 2)
        document_embeddings_t = document_embeddings.transpose(1, 2)

        query_results = []
        document_results = []

        for i,conv in enumerate(self.convolutions):
            query_conv = conv(query_embeddings_t).transpose(1, 2) 
            document_conv = conv(document_embeddings_t).transpose(1, 2)

            query_results.append(query_conv)
            document_results.append(document_conv)

        matched_results = []
        matched_results_mean = []

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        for i in range(len(query_results)):
            for t in range(len(query_results)):
                standard_res,mean_res = self.forward_matrix_kernel_pooling(query_results[i], document_results[t], query_by_doc_mask, query_pad_mask,doc_lengths)
                matched_results.append(standard_res)
                matched_results_mean.append(mean_res)

        #
        # "Learning to rank" layer
        # -------------------------------------------------------

        all_grams = torch.cat(matched_results,1)
        all_grams_mean = torch.cat(matched_results_mean,1)

        dense_out = self.dense(all_grams)
        dense_mean_out = self.dense_mean(all_grams_mean)
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out],dim=1))

        output = torch.squeeze(dense_comb_out, 1)
        return output

    #
    # create a match matrix between query & document terms
    #
    def forward_matrix_kernel_pooling(self, query_tensor, document_tensor, query_by_doc_mask, query_pad_oov_mask,doc_lengths):

        #
        # cosine matrix
        # -------------------------------------------------------
        # shape: (batch, query_max, doc_max)
        
        cosine_matrix = self.cosine_module.forward(query_tensor, document_tensor)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values

        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        per_kernel_query_mean = per_kernel_query / doc_lengths.view(-1,1,1)

        log_per_kernel_query_mean = torch.log(torch.clamp(per_kernel_query_mean, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 


        return per_kernel,per_kernel_mean

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
        l_sigma = [0.001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma


class Matchmaker_v5c(nn.Module):
    '''
    doc length avg. kernel + proximity matching from mm_light_v5c applied to conv_knrm 

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Matchmaker_v5c(word_embeddings_out_dim=word_embeddings_out_dim,
                         n_grams=config["conv_knrm_ngrams"], 
                         n_kernels=config["conv_knrm_kernels"],
                         conv_out_dim=config["conv_knrm_conv_out_dim"])

    def __init__(self,
                 word_embeddings_out_dim: int,
                 n_grams:int,
                 n_kernels: int,
                 conv_out_dim:int):

        super(Matchmaker_v5c, self).__init__()

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))

        self.convolutions = []
        for i in range(1, n_grams + 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.ConstantPad1d((0,i - 1), 0),
                    nn.Conv1d(kernel_size=i, in_channels=word_embeddings_out_dim, out_channels=conv_out_dim),
                    nn.ReLU()) 
            )
        self.convolutions = nn.ModuleList(self.convolutions) # register conv as part of the model

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        #self.pos_aware_convs = []
        self.pos_aware_linear = []
        self.pos_aware_size = []
        for i in range(2, 15):#[2,5,10,15,20,30,40,50]:#range(2, 50, 5):
            #pos_aware_conv = nn.Conv1d(kernel_size=i,in_channels=1,out_channels=1,bias=False)
            #torch.nn.init.ones_(pos_aware_conv.weight)
            #pos_aware_conv.weight.requires_grad = False
            #self.pos_aware_convs.append(
            #    nn.Sequential(
            #        nn.ConstantPad1d((0,i - 1), 0),
            #        pos_aware_conv)#,
                    #nn.ReLU()) 
            #)
            self.pos_aware_linear.append(nn.Linear(n_kernels * n_grams * n_grams, 1, bias=True))
            self.pos_aware_size.append(i)

        #self.pos_aware_convs = nn.ModuleList(self.pos_aware_convs) # register conv as part of the model
        self.pos_aware_linear = nn.ModuleList(self.pos_aware_linear) # register linears as part of the model

        self.pos_aware_combine = nn.Linear(len(self.pos_aware_linear), 1, bias=True)


        # *9 because we concat the 3x3 conv match sums together before the dense layer
        self.dense = nn.Linear(n_kernels * n_grams * n_grams, 1, bias=True) 
        self.dense_mean = nn.Linear(n_kernels * n_grams * n_grams, 1, bias=True)
        self.dense_comb = nn.Linear(3, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high fot
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo

    def get_param_stats(self):
        return "MM: dense w: "+str(self.dense.weight.data)+" b: "+str(self.dense.bias.data) +\
        "dense_mean weight: "+str(self.dense_mean.weight.data)+"b: "+str(self.dense_mean.bias.data) +\
        "dense_comb weight: "+str(self.dense_comb.weight.data) + "scaler: "+str(self.nn_scaler.data)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor) -> torch.Tensor:

        #
        # prepare embedding tensors
        # -------------------------------------------------------

        # we assume 1 is the unknown token, 0 is padding - both need to be removed
        #if len(query["tokens"].shape) == 2: # (embedding lookup matrix)
#
        #    # shape: (batch, query_max)
        #    query_pad_oov_mask = (query["tokens"] > 1).float()
        #    # shape: (batch, doc_max)
        #    document_pad_oov_mask = (document["tokens"] > 1).float()
#
        #    # shape: (batch, query_max)
        #    query_pad_mask = (query["tokens"] > 0).float()
        #    # shape: (batch, doc_max)
        #    document_pad_mask = (document["tokens"] > 0).float()
#
        #else: # == 3 (elmo characters per word)
        #    
        #    # shape: (batch, query_max)
        #    query_pad_oov_mask = (torch.sum(query["tokens"],2) > 0).float()
        #    # shape: (batch, doc_max)
        #    document_pad_oov_mask = (torch.sum(document["tokens"],2) > 0).float()
#
        ##query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)
#
        ## shape: (batch, query_max,emb_dim)
        #query_embeddings = self.word_embeddings(query) * query_pad_oov_mask.unsqueeze(-1)
        ## shape: (batch, document_max,emb_dim)
        #document_embeddings = self.word_embeddings(document) * document_pad_oov_mask.unsqueeze(-1)

        query_pad_mask = query_pad_oov_mask
        document_pad_mask = document_pad_oov_mask

        query_by_doc_mask = torch.bmm(query_pad_mask.unsqueeze(-1), document_pad_mask.unsqueeze(-1).transpose(-1, -2))

        # !! conv1d requires tensor in shape: [batch, emb_dim, sequence_length ]
        # so we transpose embedding tensors from : [batch, sequence_length,emb_dim] to [batch, emb_dim, sequence_length ]
        # feed that into the conv1d and reshape output from [batch, conv1d_out_channels, sequence_length ] 
        # to [batch, sequence_length, conv1d_out_channels]

        query_embeddings_t = query_embeddings.transpose(1, 2)
        document_embeddings_t = document_embeddings.transpose(1, 2)

        query_results = []
        document_results = []

        for i,conv in enumerate(self.convolutions):
            query_conv = conv(query_embeddings_t).transpose(1, 2) 
            document_conv = conv(document_embeddings_t).transpose(1, 2)

            query_results.append(query_conv)
            document_results.append(document_conv)

        matched_results = []
        matched_results_mean = []
        prox_results = []

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        for i in range(len(query_results)):
            for t in range(len(query_results)):
                standard_res,mean_res,prox_res = self.forward_matrix_kernel_pooling(query_results[i], document_results[t], query_by_doc_mask, query_pad_mask,doc_lengths,document_pad_oov_mask)
                matched_results.append(standard_res)
                matched_results_mean.append(mean_res)
                prox_results.append(prox_res)

        #
        # "Learning to rank" layer
        # -------------------------------------------------------

        all_grams = torch.cat(matched_results,1)
        all_grams_mean = torch.cat(matched_results_mean,1)
        all_grams_prox = torch.cat(prox_results,2)

        pos_aware_linear_result = torch.empty((query_embeddings.shape[0],len(self.pos_aware_linear)),device=query_embeddings.device)
        for i in range(len(self.pos_aware_linear)):
            pos_aware_linear_result[:,i] = self.pos_aware_linear[i](all_grams_prox[:,i]).squeeze(-1)

        pos_linear_result = self.pos_aware_combine(pos_aware_linear_result)



        dense_out = self.dense(all_grams)
        dense_mean_out = self.dense_mean(all_grams_mean)
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out,pos_linear_result],dim=1))

        output = torch.squeeze(dense_comb_out, 1)
        return output

    #
    # create a match matrix between query & document terms
    #
    def forward_matrix_kernel_pooling(self, query_tensor, document_tensor, query_by_doc_mask, query_pad_oov_mask,doc_lengths,document_pad_oov_mask):

        #
        # cosine matrix
        # -------------------------------------------------------
        # shape: (batch, query_max, doc_max)
        
        cosine_matrix = self.cosine_module.forward(query_tensor, document_tensor)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values

        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        per_kernel_query_mean = per_kernel_query / doc_lengths.view(-1,1,1)

        log_per_kernel_query_mean = torch.log(torch.clamp(per_kernel_query_mean, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 

        max_per_doc = torch.max(cosine_matrix_masked, dim=1)

        conv_res_tensor = torch.empty((cosine_matrix_extradim.shape[0],len(self.pos_aware_size),cosine_matrix_extradim.shape[2],1),device=cosine_matrix.device)
        for i,prox_window in enumerate(self.pos_aware_size):

            sliding_window_i = max_per_doc[0].unfold(dimension=1, size=prox_window, step=1)
            sliding_sums = torch.sum(sliding_window_i,dim=2) / prox_window
            sliding_sums_padded = torch.nn.functional.pad(sliding_sums,(0,prox_window-1))

            #conv_res = self.pos_aware_convs[i](max_per_doc)
            #conv_res = (conv_res.transpose(1,2) / self.pos_aware_size[i])#.unsqueeze(-1)
            conv_res_tensor[:,i] = sliding_sums_padded.unsqueeze(-1) #.view(sliding_sums_padded.shape[0],1,-1,1)

        kernel_res2 = torch.exp(- torch.pow(conv_res_tensor - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_res_masked2 = kernel_res2.squeeze() * document_pad_oov_mask.view(document_pad_oov_mask.shape[0],1,-1,1)
        kernel_res_summed2 = torch.sum(kernel_res_masked2, dim=2) * self.nn_scaler

        return per_kernel,per_kernel_mean,kernel_res_summed2

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
        l_sigma = [0.001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma


class Matchmaker_v5d(nn.Module):
    '''
    doc length avg. kernel + proximity matching from mm_light_v5d applied to conv_knrm + salience

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Matchmaker_v5d(word_embeddings_out_dim=word_embeddings_out_dim,
                         n_grams=config["conv_knrm_ngrams"], 
                         n_kernels=config["conv_knrm_kernels"],
                         conv_out_dim=config["conv_knrm_conv_out_dim"])

    def __init__(self,
                 word_embeddings_out_dim: int,
                 n_grams:int,
                 n_kernels: int,
                 conv_out_dim:int):

        super(Matchmaker_v5d, self).__init__()

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))

        self.convolutions = []
        for i in range(1, n_grams + 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.ConstantPad1d((0,i - 1), 0),
                    nn.Conv1d(kernel_size=i, in_channels=word_embeddings_out_dim, out_channels=conv_out_dim),
                    nn.ReLU()) 
            )
        self.convolutions = nn.ModuleList(self.convolutions) # register conv as part of the model

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()
        self.salc_W1 = nn.Linear(300, 1, bias=True)
        self.salc_W2 = nn.Linear(conv_out_dim, 300, bias=True)

        #self.pos_aware_convs = []
        self.pos_aware_linear = []
        self.pos_aware_size = []
        for i in range(2, 15):#[2,5,10,15,20,30,40,50]:#range(2, 50, 5):
            #pos_aware_conv = nn.Conv1d(kernel_size=i,in_channels=1,out_channels=1,bias=False)
            #torch.nn.init.ones_(pos_aware_conv.weight)
            #pos_aware_conv.weight.requires_grad = False
            #self.pos_aware_convs.append(
            #    nn.Sequential(
            #        nn.ConstantPad1d((0,i - 1), 0),
            #        pos_aware_conv)#,
                    #nn.ReLU()) 
            #)
            self.pos_aware_linear.append(nn.Linear(n_kernels * n_grams * n_grams, 1, bias=True))
            self.pos_aware_size.append(i)

        #self.pos_aware_convs = nn.ModuleList(self.pos_aware_convs) # register conv as part of the model
        self.pos_aware_linear = nn.ModuleList(self.pos_aware_linear) # register linears as part of the model

        self.pos_aware_combine = nn.Linear(len(self.pos_aware_linear), 1, bias=True)


        # *9 because we concat the 3x3 conv match sums together before the dense layer
        self.dense = nn.Linear(n_kernels * n_grams * n_grams, 1, bias=True) 
        self.dense_mean = nn.Linear(n_kernels * n_grams * n_grams, 1, bias=True)
        self.dense_comb = nn.Linear(3, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high fot
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo

    def get_param_stats(self):
        return "MM: dense w: "+str(self.dense.weight.data)+" b: "+str(self.dense.bias.data) +\
        "dense_mean weight: "+str(self.dense_mean.weight.data)+"b: "+str(self.dense_mean.bias.data) +\
        "dense_comb weight: "+str(self.dense_comb.weight.data) + "scaler: "+str(self.nn_scaler.data)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor) -> torch.Tensor:

        #
        # prepare embedding tensors
        # -------------------------------------------------------

        # we assume 1 is the unknown token, 0 is padding - both need to be removed
        #if len(query["tokens"].shape) == 2: # (embedding lookup matrix)
#
        #    # shape: (batch, query_max)
        #    query_pad_oov_mask = (query["tokens"] > 1).float()
        #    # shape: (batch, doc_max)
        #    document_pad_oov_mask = (document["tokens"] > 1).float()
#
        #    # shape: (batch, query_max)
        #    query_pad_mask = (query["tokens"] > 0).float()
        #    # shape: (batch, doc_max)
        #    document_pad_mask = (document["tokens"] > 0).float()
#
        #else: # == 3 (elmo characters per word)
        #    
        #    # shape: (batch, query_max)
        #    query_pad_oov_mask = (torch.sum(query["tokens"],2) > 0).float()
        #    # shape: (batch, doc_max)
        #    document_pad_oov_mask = (torch.sum(document["tokens"],2) > 0).float()
#
        ##query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)
#
        ## shape: (batch, query_max,emb_dim)
        #query_embeddings = self.word_embeddings(query) * query_pad_oov_mask.unsqueeze(-1)
        ## shape: (batch, document_max,emb_dim)
        #document_embeddings = self.word_embeddings(document) * document_pad_oov_mask.unsqueeze(-1)

        query_pad_mask = query_pad_oov_mask
        document_pad_mask = document_pad_oov_mask

        query_by_doc_mask = torch.bmm(query_pad_mask.unsqueeze(-1), document_pad_mask.unsqueeze(-1).transpose(-1, -2))

        # !! conv1d requires tensor in shape: [batch, emb_dim, sequence_length ]
        # so we transpose embedding tensors from : [batch, sequence_length,emb_dim] to [batch, emb_dim, sequence_length ]
        # feed that into the conv1d and reshape output from [batch, conv1d_out_channels, sequence_length ] 
        # to [batch, sequence_length, conv1d_out_channels]

        query_embeddings_t = query_embeddings.transpose(1, 2)
        document_embeddings_t = document_embeddings.transpose(1, 2)

        query_results = []
        document_results = []

        for i,conv in enumerate(self.convolutions):
            query_conv = conv(query_embeddings_t).transpose(1, 2) 
            document_conv = conv(document_embeddings_t).transpose(1, 2)

            query_results.append(query_conv)
            document_results.append(document_conv)

        matched_results = []
        matched_results_mean = []
        prox_results = []

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        for i in range(len(query_results)):
            for t in range(len(query_results)):
                standard_res,mean_res,prox_res = self.forward_matrix_kernel_pooling(query_results[i], document_results[t], query_by_doc_mask, query_pad_mask,doc_lengths,document_pad_oov_mask)
                matched_results.append(standard_res)
                matched_results_mean.append(mean_res)
                prox_results.append(prox_res)

        #
        # "Learning to rank" layer
        # -------------------------------------------------------

        all_grams = torch.cat(matched_results,1)
        all_grams_mean = torch.cat(matched_results_mean,1)
        all_grams_prox = torch.cat(prox_results,2)

        pos_aware_linear_result = torch.empty((query_embeddings.shape[0],len(self.pos_aware_linear)),device=query_embeddings.device)
        for i in range(len(self.pos_aware_linear)):
            pos_aware_linear_result[:,i] = self.pos_aware_linear[i](all_grams_prox[:,i]).squeeze(-1)

        pos_linear_result = self.pos_aware_combine(pos_aware_linear_result)



        dense_out = self.dense(all_grams)
        dense_mean_out = self.dense_mean(all_grams_mean)
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out,pos_linear_result],dim=1))

        output = torch.squeeze(dense_comb_out, 1)
        return output

    #
    # create a match matrix between query & document terms
    #
    def forward_matrix_kernel_pooling(self, query_tensor, document_tensor, query_by_doc_mask, query_pad_oov_mask,doc_lengths,document_pad_oov_mask):

        #
        # cosine matrix
        # -------------------------------------------------------
        # shape: (batch, query_max, doc_max)
        
        cosine_matrix = self.cosine_module.forward(query_tensor, document_tensor)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        #cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        qry_salc = self.salc_W1(torch.tanh(self.salc_W2(query_tensor)))
        doc_salc = self.salc_W1(torch.tanh(self.salc_W2(document_tensor)))
        
        qry_salc_matrix = qry_salc.repeat(1, 1, doc_salc.size()[1])
        doc_salc_matrix = doc_salc.repeat(1, 1, qry_salc.size()[1]).transpose(2,1)
        
        salc_matrix = torch.sigmoid(qry_salc_matrix) * torch.sigmoid(doc_salc_matrix)
        
        # match matrix
        match_matrix_masked = torch.tanh(cosine_matrix_masked * salc_matrix)
        cosine_matrix_extradim = match_matrix_masked.unsqueeze(-1)


        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values

        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        per_kernel_query_mean = per_kernel_query / doc_lengths.view(-1,1,1)

        log_per_kernel_query_mean = torch.log(torch.clamp(per_kernel_query_mean, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 

        max_per_doc = torch.max(cosine_matrix_masked, dim=1)

        conv_res_tensor = torch.empty((cosine_matrix_extradim.shape[0],len(self.pos_aware_size),cosine_matrix_extradim.shape[2],1),device=cosine_matrix.device)
        for i,prox_window in enumerate(self.pos_aware_size):

            sliding_window_i = max_per_doc[0].unfold(dimension=1, size=prox_window, step=1)
            sliding_sums = torch.sum(sliding_window_i,dim=2) / prox_window
            sliding_sums_padded = torch.nn.functional.pad(sliding_sums,(0,prox_window-1))

            #conv_res = self.pos_aware_convs[i](max_per_doc)
            #conv_res = (conv_res.transpose(1,2) / self.pos_aware_size[i])#.unsqueeze(-1)
            conv_res_tensor[:,i] = sliding_sums_padded.unsqueeze(-1) #.view(sliding_sums_padded.shape[0],1,-1,1)

        kernel_res2 = torch.exp(- torch.pow(conv_res_tensor - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_res_masked2 = kernel_res2.squeeze() * document_pad_oov_mask.view(document_pad_oov_mask.shape[0],1,-1,1)
        kernel_res_summed2 = torch.sum(kernel_res_masked2, dim=2) * self.nn_scaler

        return per_kernel,per_kernel_mean,kernel_res_summed2

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
        l_sigma = [0.001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma



class Matchmaker_v6a(nn.Module):
    '''
    doc length avg. kernel + proximity matching from mm_light_v6a applied to conv_knrm + explicit salience

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim,vocabulary_size):
        return Matchmaker_v6a(vocabulary_size=vocabulary_size,
                         word_embeddings_out_dim=word_embeddings_out_dim,
                         n_grams=config["conv_knrm_ngrams"], 
                         n_kernels=config["conv_knrm_kernels"],
                         conv_out_dim=config["conv_knrm_conv_out_dim"])

    def __init__(self,
                 vocabulary_size:int,
                 word_embeddings_out_dim: int,
                 n_grams:int,
                 n_kernels: int,
                 conv_out_dim:int):

        super(Matchmaker_v6a, self).__init__()

        self.salience_weights = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=1, padding_idx=0)
        torch.nn.init.constant_(self.salience_weights.weight,1)


        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))

        self.convolutions = []
        self.salience_pools = []
        for i in range(1, n_grams + 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.ConstantPad1d((0,i - 1), 0),
                    nn.Conv1d(kernel_size=i, in_channels=word_embeddings_out_dim, out_channels=conv_out_dim),
                    nn.ReLU()) 
            )
            self.salience_pools.append(nn.Sequential(
                    nn.ConstantPad1d((0,i - 1), 0),
                    nn.AvgPool1d(kernel_size=i)
            ))
        self.salience_pools = nn.ModuleList(self.salience_pools)
        self.convolutions = nn.ModuleList(self.convolutions) # register conv as part of the model

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()
        #self.salc_W1 = nn.Linear(300, 1, bias=True)
        #self.salc_W2 = nn.Linear(conv_out_dim, 300, bias=True)

        #self.pos_aware_convs = []
        self.pos_aware_linear = []
        self.pos_aware_size = []
        for i in range(2, 15):#[2,5,10,15,20,30,40,50]:#range(2, 50, 5):
            #pos_aware_conv = nn.Conv1d(kernel_size=i,in_channels=1,out_channels=1,bias=False)
            #torch.nn.init.ones_(pos_aware_conv.weight)
            #pos_aware_conv.weight.requires_grad = False
            #self.pos_aware_convs.append(
            #    nn.Sequential(
            #        nn.ConstantPad1d((0,i - 1), 0),
            #        pos_aware_conv)#,
                    #nn.ReLU()) 
            #)
            self.pos_aware_linear.append(nn.Linear(n_kernels * n_grams * n_grams, 1, bias=True))
            self.pos_aware_size.append(i)

        #self.pos_aware_convs = nn.ModuleList(self.pos_aware_convs) # register conv as part of the model
        self.pos_aware_linear = nn.ModuleList(self.pos_aware_linear) # register linears as part of the model

        self.pos_aware_combine = nn.Linear(len(self.pos_aware_linear), 1, bias=True)


        # *9 because we concat the 3x3 conv match sums together before the dense layer
        self.dense = nn.Linear(n_kernels * n_grams * n_grams, 1, bias=True) 
        self.dense_mean = nn.Linear(n_kernels * n_grams * n_grams, 1, bias=True)
        self.dense_comb = nn.Linear(3, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high fot
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo

    def get_param_stats(self):
        return "MM: dense w: "+str(self.dense.weight.data)+" b: "+str(self.dense.bias.data) +\
        "dense_mean weight: "+str(self.dense_mean.weight.data)+"b: "+str(self.dense_mean.bias.data) +\
        "dense_comb weight: "+str(self.dense_comb.weight.data) + "scaler: "+str(self.nn_scaler.data)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor,
                query_ids: torch.Tensor, document_ids: torch.Tensor) -> torch.Tensor:

        #
        # prepare embedding tensors
        # -------------------------------------------------------

        query_salience = torch.sigmoid(self.salience_weights(query_ids))
        document_salience = torch.sigmoid(self.salience_weights(document_ids))


        query_pad_mask = query_pad_oov_mask
        document_pad_mask = document_pad_oov_mask

        query_by_doc_mask = torch.bmm(query_pad_mask.unsqueeze(-1), document_pad_mask.unsqueeze(-1).transpose(-1, -2))

        # !! conv1d requires tensor in shape: [batch, emb_dim, sequence_length ]
        # so we transpose embedding tensors from : [batch, sequence_length,emb_dim] to [batch, emb_dim, sequence_length ]
        # feed that into the conv1d and reshape output from [batch, conv1d_out_channels, sequence_length ] 
        # to [batch, sequence_length, conv1d_out_channels]

        query_embeddings_t = query_embeddings.transpose(1, 2)
        document_embeddings_t = document_embeddings.transpose(1, 2)

        query_results = []
        document_results = []
        query_results_sal = []
        document_results_sal = []

        for i,conv in enumerate(self.convolutions):
            query_results.append(conv(query_embeddings_t).transpose(1, 2))
            document_results.append(conv(document_embeddings_t).transpose(1, 2))

        for i,pool in enumerate(self.salience_pools):
            query_results_sal.append(pool(query_salience))
            document_results_sal.append(pool(document_salience))


        matched_results = []
        matched_results_mean = []
        prox_results = []

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        for i in range(len(query_results)):
            for t in range(len(query_results)):
                standard_res,mean_res,prox_res = self.forward_matrix_kernel_pooling(query_results[i], document_results[t], query_results_sal[i], document_results_sal[t], query_by_doc_mask, query_pad_mask,doc_lengths,document_pad_oov_mask)
                matched_results.append(standard_res)
                matched_results_mean.append(mean_res)
                prox_results.append(prox_res)

        #
        # "Learning to rank" layer
        # -------------------------------------------------------

        all_grams = torch.cat(matched_results,1)
        all_grams_mean = torch.cat(matched_results_mean,1)
        all_grams_prox = torch.cat(prox_results,2)

        pos_aware_linear_result = torch.empty((query_embeddings.shape[0],len(self.pos_aware_linear)),device=query_embeddings.device)
        for i in range(len(self.pos_aware_linear)):
            pos_aware_linear_result[:,i] = self.pos_aware_linear[i](all_grams_prox[:,i]).squeeze(-1)

        pos_linear_result = self.pos_aware_combine(pos_aware_linear_result)



        dense_out = self.dense(all_grams)
        dense_mean_out = self.dense_mean(all_grams_mean)
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out,pos_linear_result],dim=1))

        output = torch.squeeze(dense_comb_out, 1)
        return output

    #
    # create a match matrix between query & document terms
    #
    def forward_matrix_kernel_pooling(self, query_tensor, document_tensor,query_sal, document_sal, query_by_doc_mask, query_pad_oov_mask,doc_lengths,document_pad_oov_mask):

        #
        # cosine matrix
        # -------------------------------------------------------
        # shape: (batch, query_max, doc_max)
        query_by_doc_salience = torch.bmm(query_sal, document_sal.transpose(-1, -2))

        cosine_matrix = self.cosine_module.forward(query_tensor, document_tensor)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        #cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        
        # match matrix
        match_matrix_masked = torch.tanh(cosine_matrix_masked * query_by_doc_salience)
        cosine_matrix_extradim = match_matrix_masked.unsqueeze(-1)


        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values

        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        per_kernel_query_mean = per_kernel_query / doc_lengths.view(-1,1,1)

        log_per_kernel_query_mean = torch.log(torch.clamp(per_kernel_query_mean, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 

        max_per_doc = torch.max(cosine_matrix_masked, dim=1)

        conv_res_tensor = torch.empty((cosine_matrix_extradim.shape[0],len(self.pos_aware_size),cosine_matrix_extradim.shape[2],1),device=cosine_matrix.device)
        for i,prox_window in enumerate(self.pos_aware_size):

            sliding_window_i = max_per_doc[0].unfold(dimension=1, size=prox_window, step=1)
            sliding_sums = torch.sum(sliding_window_i,dim=2) / prox_window
            sliding_sums_padded = torch.nn.functional.pad(sliding_sums,(0,prox_window-1))

            #conv_res = self.pos_aware_convs[i](max_per_doc)
            #conv_res = (conv_res.transpose(1,2) / self.pos_aware_size[i])#.unsqueeze(-1)
            conv_res_tensor[:,i] = sliding_sums_padded.unsqueeze(-1) #.view(sliding_sums_padded.shape[0],1,-1,1)

        kernel_res2 = torch.exp(- torch.pow(conv_res_tensor - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_res_masked2 = kernel_res2.squeeze() * document_pad_oov_mask.view(document_pad_oov_mask.shape[0],1,-1,1)
        kernel_res_summed2 = torch.sum(kernel_res_masked2, dim=2) * self.nn_scaler

        return per_kernel,per_kernel_mean,kernel_res_summed2

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
        l_sigma = [0.001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma
