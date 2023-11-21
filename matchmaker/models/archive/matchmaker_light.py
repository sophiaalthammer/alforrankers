from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          
from allennlp.modules.matrix_attention.dot_product_matrix_attention import *                          

class Matchmaker_light_v1(nn.Module):
    '''
    Paper: ...

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Matchmaker_light_v1(n_kernels = config["mm_light_kernels"])

    def __init__(self,
                 n_kernels: int):

        super(Matchmaker_light_v1, self).__init__()

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels * 2, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------

        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #
        # mean kernels
        #
        #kernel_results_masked2 = kernel_results_masked.clone()

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        #kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * 0.01
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        #per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)

        per_kernel_query_mean = per_kernel_query / doc_lengths.view(-1,1,1)

        log_per_kernel_query_mean = torch.log(torch.clamp(per_kernel_query_mean, min=1e-10)) * 0.01
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 


        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(torch.cat([per_kernel,per_kernel_mean],dim=1))
        score = torch.squeeze(dense_out,1) #torch.tanh(dense_out), 1)
        return score

    def get_param_stats(self):
        return "MM_light: linear weight: "+str(self.dense.weight.data)

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

        l_sigma += [bin_size] * (n_kernels - 1) # BETTER without 0.5
        return l_sigma

class Matchmaker_light_v1b(nn.Module):
    '''
    Paper: ...

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Matchmaker_light_v1b(n_kernels = config["mm_light_kernels"])

    def __init__(self,
                 n_kernels: int):

        super(Matchmaker_light_v1b, self).__init__()

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels * 2, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------

        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #
        # mean kernels
        #
        #kernel_results_masked2 = kernel_results_masked.clone()

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        #kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * 0.01
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        #per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)

        per_kernel_query_mean = per_kernel_query / doc_lengths.view(-1,1,1)

        log_per_kernel_query_mean = torch.log(1 + torch.clamp(per_kernel_query_mean, min=1e-10)) * 0.01
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 

        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(torch.cat([per_kernel,per_kernel_mean],dim=1))
        score = torch.squeeze(dense_out,1) #torch.tanh(dense_out), 1)
        return score

    def get_param_stats(self):
        return "MM_light: linear weight: "+str(self.dense.weight.data)

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


class Matchmaker_light_v1c(nn.Module):
    '''
    Idea: have 2 sets of kernels: 1 standard (summed matches) + 1 of (average matches - based on doc length) -> combine the 2 output values weighted for final score
    -> so now we also have a component that takes the doc length into account
    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Matchmaker_light_v1c(n_kernels = config["mm_light_kernels"])

    def __init__(self,
                 n_kernels: int):

        super(Matchmaker_light_v1c, self).__init__()

        # static - kernel size & magnitude variables
        self.mu = nn.Parameter(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = nn.Parameter(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=True)
        self.dense_mean = nn.Linear(n_kernels, 1, bias=True)
        self.dense_comb = nn.Linear(2, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        torch.nn.init.uniform_(self.dense_mean.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------

        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #
        # mean kernels
        #
        #kernel_results_masked2 = kernel_results_masked.clone()

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        #kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        #per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)

        per_kernel_query_mean = per_kernel_query / doc_lengths.view(-1,1,1)

        log_per_kernel_query_mean = torch.log(torch.clamp(per_kernel_query_mean, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 

        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        dense_mean_out = self.dense_mean(per_kernel_mean)
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out],dim=1))
        score = torch.squeeze(dense_comb_out,1) #torch.tanh(dense_out), 1)
        return score

    def get_param_stats(self):
        return "MM_light: dense w: "+str(self.dense.weight.data)+" b: "+str(self.dense.bias.data) +\
        "dense_mean weight: "+str(self.dense_mean.weight.data)+"b: "+str(self.dense_mean.bias.data) +\
        "dense_comb weight: "+str(self.dense_comb.weight.data) + "scaler: "+str(self.nn_scaler.data)

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

        l_sigma += [bin_size] * (n_kernels - 1)
        return l_sigma

class Matchmaker_light_v1d(nn.Module):
    '''
    Idea: have 2 sets of kernels: 1 standard (summed matches) + 1 of (average matches - based on doc length) -> combine the 2 output values weighted for final score
    -> so now we also have a component that takes the doc length into account

    + salience from salc_knrm

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Matchmaker_light_v1d(word_embsize=word_embeddings_out_dim,
                                    n_kernels = config["mm_light_kernels"])

    def __init__(self,
                word_embsize:int,
                 n_kernels: int):

        super(Matchmaker_light_v1d, self).__init__()

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        #weights for term saliency
        self.salc_W1 = nn.Linear(300, 1, bias=True)
        self.salc_W2 = nn.Linear(word_embsize, 300, bias=True)
        

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=True)
        self.dense_mean = nn.Linear(n_kernels, 1, bias=True)
        self.dense_comb = nn.Linear(2, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        torch.nn.init.uniform_(self.dense_mean.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------



        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        #cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        qry_salc = self.salc_W1(torch.tanh(self.salc_W2(query_embeddings)))
        doc_salc = self.salc_W1(torch.tanh(self.salc_W2(document_embeddings)))
        
        qry_salc_matrix = qry_salc.repeat(1, 1, doc_salc.size()[1])
        doc_salc_matrix = doc_salc.repeat(1, 1, qry_salc.size()[1]).transpose(2,1)
        
        salc_matrix = torch.sigmoid(qry_salc_matrix) * torch.sigmoid(doc_salc_matrix)

        match_matrix_masked = torch.tanh(cosine_matrix_masked * salc_matrix)
        match_matrix_extradim = match_matrix_masked.unsqueeze(-1)


        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(match_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #
        # mean kernels
        #
        #kernel_results_masked2 = kernel_results_masked.clone()

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        #kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        #per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)

        per_kernel_query_mean = per_kernel_query / doc_lengths.view(-1,1,1)

        log_per_kernel_query_mean = torch.log(torch.clamp(per_kernel_query_mean, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 

        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        dense_mean_out = self.dense_mean(per_kernel_mean)
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out],dim=1))
        score = torch.squeeze(dense_comb_out,1) #torch.tanh(dense_out), 1)
        return score

    def get_param_stats(self):
        return "MM_light: dense w: "+str(self.dense.weight.data)+" b: "+str(self.dense.bias.data) +\
        "dense_mean weight: "+str(self.dense_mean.weight.data)+"b: "+str(self.dense_mean.bias.data) +\
        "dense_comb weight: "+str(self.dense_comb.weight.data) + "scaler: "+str(self.nn_scaler.data)

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

        l_sigma += [bin_size] * (n_kernels - 1)
        return l_sigma

class Matchmaker_light_v2(nn.Module):
    '''
    Paper: ...

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Matchmaker_light_v2(word_embeddings_out_dim, n_kernels = config["mm_light_kernels"])

    def __init__(self,
                 _embsize:int,
                 n_kernels: int):

        super(Matchmaker_light_v2, self).__init__()

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        self.salc_W1 = nn.Linear(300, 1, bias=True)
        self.salc_W2 = nn.Linear(_embsize, 300, bias=True)



        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------

        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #
        # mean kernels
        #
        #kernel_results_masked2 = kernel_results_masked.clone()

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        #kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)

        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * 0.01
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values


        qry_salc = self.salc_W1(torch.tanh(self.salc_W2(query_embeddings)))

        qry_salc = torch.sigmoid(qry_salc)

        log_per_kernel_query_masked_weighted = log_per_kernel_query_masked * qry_salc

        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 


        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        score = torch.squeeze(dense_out,1) #torch.tanh(dense_out), 1)
        return score

    def get_param_stats(self):
        return "MM_light: linear weight: "+str(self.dense.weight.data)

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

#from https://gist.github.com/kaniblu/94f3ede72d1651b087a561cf80b306ca thanks!
class MaskedSoftmax(nn.Module):
    def __init__(self):
        super(MaskedSoftmax, self).__init__()
        self.softmax = nn.Softmax(1)

    def forward(self, x, mask=None):
        """
        Performs masked softmax, as simply masking post-softmax can be
        inaccurate
        :param x: [batch_size, num_items]
        :param mask: [batch_size, num_items]
        :return:
        """
        if mask is not None:
            mask = mask.float()
        if mask is not None:
            x_masked = x * mask + (1 - 1 / mask)
        else:
            x_masked = x
        x_max = x_masked.max(1)[0]
        x_exp = (x - x_max.unsqueeze(-1)).exp()
        if mask is not None:
            x_exp = x_exp * mask.float()
        return x_exp / x_exp.sum(1).unsqueeze(-1)


class Matchmaker_light_v3(nn.Module):
    '''
    Paper: ...

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Matchmaker_light_v3(word_embeddings_out_dim, n_kernels = config["mm_light_kernels"])

    def __init__(self,
                 _embsize:int,
                 n_kernels: int):

        super(Matchmaker_light_v3, self).__init__()

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        self.salc_W1 = nn.Linear(300, 1, bias=True)
        self.salc_W2 = nn.Linear(_embsize, 300, bias=True)

        self.salc_softmax = MaskedSoftmax()

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------

        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #
        # mean kernels
        #
        #kernel_results_masked2 = kernel_results_masked.clone()

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        #kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)

        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * 0.01
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values


        qry_salc = self.salc_W1(torch.tanh(self.salc_W2(query_embeddings)))

        qry_salc = self.salc_softmax(qry_salc,query_pad_oov_mask.unsqueeze(-1))

        log_per_kernel_query_masked_weighted = log_per_kernel_query_masked * qry_salc

        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 


        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        score = torch.squeeze(dense_out,1) #torch.tanh(dense_out), 1)
        return score

    def get_param_stats(self):
        return "MM_light: linear weight: "+str(self.dense.weight.data)

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

class Matchmaker_light_v3b(nn.Module):
    '''
    Paper: ...

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Matchmaker_light_v3b(word_embeddings_out_dim, n_kernels = config["mm_light_kernels"])

    def __init__(self,
                 _embsize:int,
                 n_kernels: int):

        super(Matchmaker_light_v3b, self).__init__()

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))


        self.salc_att = MultiHeadSelfAttention(num_heads=16,
                 input_dim=_embsize,
                 attention_dim=64,
                 values_dim=64,
                 output_projection_dim=64,
                 attention_dropout_prob=0.2)

        self.salc_W1 = nn.Linear(64, 64, bias=True)
        self.salc_W2 = nn.Linear(64, 1, bias=True)

        self.salc_softmax = MaskedSoftmax()

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------

        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #
        # mean kernels
        #
        #kernel_results_masked2 = kernel_results_masked.clone()

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        #kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        qry_salc = self.salc_att(query_embeddings,query_pad_oov_mask)
        qry_salc = torch.sigmoid(self.salc_W2(torch.tanh(self.salc_W1(qry_salc))))

        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query * qry_salc, min=1e-10))*self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values


        #qry_salc = self.salc_W1(torch.tanh(self.salc_W2(query_embeddings)))


        #qry_salc = self.salc_softmax(qry_salc,query_pad_oov_mask.unsqueeze(-1))

        log_per_kernel_query_masked_weighted = log_per_kernel_query_masked * qry_salc

        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 


        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        score = torch.squeeze(dense_out,1) #torch.tanh(dense_out), 1)
        return score

    def get_param_stats(self):
        return "MM_light: linear weight: "+str(self.dense.weight.data)+ "scaler: "+str(self.nn_scaler.data)

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

class Matchmaker_light_v4(nn.Module):
    '''
    Paper: ...

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Matchmaker_light_v4(word_embeddings_out_dim, n_kernels = config["mm_light_kernels"])

    def __init__(self,
                 _embsize:int,
                 n_kernels: int):

        super(Matchmaker_light_v4, self).__init__()

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))

        self.stacked_att = StackedSelfAttentionEncoder(input_dim=_embsize,
                 hidden_dim=_embsize,
                 projection_dim=32,
                 feedforward_hidden_dim=100,
                 num_layers=1,
                 num_attention_heads=32,
                 dropout_prob = 0,
                 residual_dropout_prob = 0,
                 attention_dropout_prob= 0,
                 #use_positional_encoding=False
                 )

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        #self.salc_W1 = nn.Linear(100, 1, bias=True)
        #self.salc_W2 = nn.Linear(_embsize, 100, bias=True)

        #self.salc_softmax = MaskedSoftmax()

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=True)
        self.dense_mean = nn.Linear(n_kernels, 1, bias=True)
        self.dense_comb = nn.Linear(2, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        torch.nn.init.uniform_(self.dense_mean.weight, -0.014, 0.014)  # inits taken from matchzoo

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ

        query_embeddings = query_embeddings * query_pad_oov_mask.unsqueeze(-1) #* 10
        document_embeddings=document_embeddings * document_pad_oov_mask.unsqueeze(-1) #* 10

        query_embeddings_context = self.stacked_att(query_embeddings,query_pad_oov_mask)
        document_embeddings_context = self.stacked_att(document_embeddings,document_pad_oov_mask)

        query_embeddings = torch.cat([query_embeddings,query_embeddings_context],dim=2) * query_pad_oov_mask.unsqueeze(-1)
        document_embeddings = torch.cat([document_embeddings,document_embeddings_context],dim=2) * document_pad_oov_mask.unsqueeze(-1)

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------


        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = torch.tanh(cosine_matrix * query_by_doc_mask)
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #
        # mean kernels
        #
        #kernel_results_masked2 = kernel_results_masked.clone()

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        #kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        #per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)

        per_kernel_query_mean = per_kernel_query / doc_lengths.view(-1,1,1)

        log_per_kernel_query_mean = torch.log(torch.clamp(per_kernel_query_mean, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 


        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        dense_mean_out = self.dense_mean(per_kernel_mean)
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out],dim=1))
        score = torch.squeeze(dense_comb_out,1) #torch.tanh(dense_out), 1)
        return score

    def get_param_stats(self):
        return "MM_light_v4: dense w: "+str(self.dense.weight.data)+" b: "+str(self.dense.bias.data) +\
        "dense_mean weight: "+str(self.dense_mean.weight.data)+"b: "+str(self.dense_mean.bias.data) +\
        "dense_comb weight: "+str(self.dense_comb.weight.data) + "scaler: "+str(self.nn_scaler.data)

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

class Matchmaker_light_v4b(nn.Module):
    '''
    Paper: ...

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Matchmaker_light_v4b(word_embeddings_out_dim, n_kernels = config["mm_light_kernels"])

    def __init__(self,
                 _embsize:int,
                 n_kernels: int):

        super(Matchmaker_light_v4b, self).__init__()

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))
        self.mixer = nn.Parameter(torch.full([1,1,1], 0.5, dtype=torch.float32, requires_grad=True))

        self.stacked_att = StackedSelfAttentionEncoder(input_dim=_embsize,
                 hidden_dim=_embsize,
                 projection_dim=32,
                 feedforward_hidden_dim=100,
                 num_layers=2,
                 num_attention_heads=16,
                 dropout_prob = 0,
                 residual_dropout_prob = 0,
                 attention_dropout_prob= 0,
                 #use_positional_encoding=False
                 )

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        #self.salc_W1 = nn.Linear(100, 1, bias=True)
        #self.salc_W2 = nn.Linear(_embsize, 100, bias=True)

        #self.salc_softmax = MaskedSoftmax()

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=True)
        self.dense_mean = nn.Linear(n_kernels, 1, bias=True)
        self.dense_comb = nn.Linear(2, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        torch.nn.init.uniform_(self.dense_mean.weight, -0.014, 0.014)  # inits taken from matchzoo

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ

        query_embeddings = query_embeddings * query_pad_oov_mask.unsqueeze(-1)
        document_embeddings = document_embeddings * document_pad_oov_mask.unsqueeze(-1)

        query_embeddings_context = self.stacked_att(query_embeddings,query_pad_oov_mask)
        document_embeddings_context = self.stacked_att(document_embeddings,document_pad_oov_mask)

        #query_embeddings = torch.cat([query_embeddings,query_embeddings_context],dim=2) * query_pad_oov_mask.unsqueeze(-1)
        #document_embeddings = torch.cat([document_embeddings,document_embeddings_context],dim=2) * document_pad_oov_mask.unsqueeze(-1)
        query_embeddings = (self.mixer * query_embeddings + (1 - self.mixer) * query_embeddings_context) * query_pad_oov_mask.unsqueeze(-1)
        document_embeddings = (self.mixer * document_embeddings + (1 - self.mixer) * document_embeddings_context) * document_pad_oov_mask.unsqueeze(-1)

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------


        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = torch.tanh(cosine_matrix * query_by_doc_mask)
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #
        # mean kernels
        #
        #kernel_results_masked2 = kernel_results_masked.clone()

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        #kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        #per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)

        per_kernel_query_mean = per_kernel_query / (doc_lengths.view(-1,1,1) + 1) # well, that +1 needs an explanation, sometimes training data is just broken ... (and nans all the things!)

        log_per_kernel_query_mean = torch.log(torch.clamp(per_kernel_query_mean, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 


        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        dense_mean_out = self.dense_mean(per_kernel_mean)
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out],dim=1))
        score = torch.squeeze(dense_comb_out,1) #torch.tanh(dense_out), 1)
        return score

    def get_param_stats(self):
        return "MM_light_v4b: dense w: "+str(self.dense.weight.data)+" b: "+str(self.dense.bias.data) +\
        "dense_mean weight: "+str(self.dense_mean.weight.data)+"b: "+str(self.dense_mean.bias.data) +\
        "dense_comb weight: "+str(self.dense_comb.weight.data) + "scaler: "+str(self.nn_scaler.data) +"mixer: "+str(self.mixer.data)

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


class Matchmaker_light_v4c(nn.Module):
    '''
    Paper: ...

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Matchmaker_light_v4c(word_embeddings_out_dim, n_kernels = config["mm_light_kernels"])

    def __init__(self,
                 _embsize:int,
                 n_kernels: int):

        super(Matchmaker_light_v4c, self).__init__()

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))
        self.mixer = nn.Parameter(torch.full([1,1,1], 0.5, dtype=torch.float32, requires_grad=True))

        self.stacked_att = StackedSelfAttentionEncoder(input_dim=_embsize,
                 hidden_dim=_embsize,
                 projection_dim=32,
                 feedforward_hidden_dim=100,
                 num_layers=2,
                 num_attention_heads=16,
                 dropout_prob = 0.1,
                 residual_dropout_prob = 0.1,
                 attention_dropout_prob= 0.1,
                 #use_positional_encoding=False
                 )

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        #self.salc_W1 = nn.Linear(100, 1, bias=True)
        #self.salc_W2 = nn.Linear(_embsize, 100, bias=True)

        #self.salc_softmax = MaskedSoftmax()

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=True)
        self.dense_mean = nn.Linear(n_kernels, 1, bias=True)
        self.dense_euclidean_dist = nn.Linear(n_kernels, 1, bias=True)
        self.dense_comb = nn.Linear(3, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        torch.nn.init.uniform_(self.dense_mean.weight, -0.014, 0.014)  # inits taken from matchzoo

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ

        query_embeddings = query_embeddings * query_pad_oov_mask.unsqueeze(-1) #* 10
        document_embeddings=document_embeddings * document_pad_oov_mask.unsqueeze(-1) #* 10

        query_embeddings_context = self.stacked_att(query_embeddings,query_pad_oov_mask)
        document_embeddings_context = self.stacked_att(document_embeddings,document_pad_oov_mask)

        query_embeddings = (self.mixer * query_embeddings + (1 - self.mixer) * query_embeddings_context) * query_pad_oov_mask.unsqueeze(-1)
        document_embeddings = (self.mixer * document_embeddings + (1 - self.mixer) * document_embeddings_context) * document_pad_oov_mask.unsqueeze(-1)

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------


        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = torch.tanh(cosine_matrix * query_by_doc_mask)
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #
        # mean kernels
        #
        #kernel_results_masked2 = kernel_results_masked.clone()

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        #kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        #per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)

        per_kernel_query_mean = per_kernel_query / doc_lengths.view(-1,1,1)

        log_per_kernel_query_mean = torch.log(torch.clamp(per_kernel_query_mean, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 

        #
        # euclid
        #
        #query_norm = query_embeddings/torch.clamp(torch.norm(query_embeddings,dim=-1).unsqueeze(-1),min=1e-10)
        #doc_norm = document_embeddings/torch.clamp(torch.norm(document_embeddings,dim=-1).unsqueeze(-1),min=1e-10)

        euclidean_matrix = torch.sqrt(torch.norm(query_embeddings.unsqueeze(2) - document_embeddings.unsqueeze(1),p=2,dim=-1))
        #euclidean_matrix = torch.norm(query_embeddings.unsqueeze(2) - document_embeddings.unsqueeze(1),p=2,dim=-1)
        euclidean_matrix_masked = torch.tanh(euclidean_matrix * query_by_doc_mask) - 1
        euclidean_matrix_extradim = euclidean_matrix_masked.unsqueeze(-1)

        raw_kernel_results = torch.exp(- torch.pow(euclidean_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_euclid = torch.sum(log_per_kernel_query_masked, 1) 



        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        dense_mean_out = self.dense_mean(per_kernel_mean)
        dense_euclidean_dist_out = self.dense_euclidean_dist(per_kernel_euclid)
        
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out,dense_euclidean_dist_out],dim=1))
        score = torch.squeeze(dense_euclidean_dist_out,1) #torch.tanh(dense_out), 1)
        return score

    def get_param_stats(self):
        return "MM_light_v4b: dense w: "+str(self.dense.weight.data)+" b: "+str(self.dense.bias.data) +\
        "dense_mean weight: "+str(self.dense_mean.weight.data)+"b: "+str(self.dense_mean.bias.data) +\
        "dense_comb weight: "+str(self.dense_comb.weight.data) + "scaler: "+str(self.nn_scaler.data) +"mixer: "+str(self.mixer.data)

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

class Matchmaker_light_v5(nn.Module):
    '''
    Paper: ...

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Matchmaker_light_v5(n_kernels = config["mm_light_kernels"])

    def __init__(self,
                 n_kernels: int):

        super(Matchmaker_light_v5, self).__init__()

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        pos_awareness = nn.Conv1d(kernel_size=4,in_channels=1,out_channels=1,bias=False)
        torch.nn.init.ones_(pos_awareness.weight)
        #pos_awareness.weight.requires_grad = False

        self.pos_awareness = nn.Sequential(
                    nn.ConstantPad1d((0,4 - 1), 0),
                    pos_awareness)

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=True)
        self.dense_mean = nn.Linear(n_kernels, 1, bias=True)

        self.dense_pos_aware = nn.Linear(n_kernels, 1, bias=True)

        self.dense_comb = nn.Linear(3, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        torch.nn.init.uniform_(self.dense_mean.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------

        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # positional awareness
        #
        query_lengths = torch.sum(query_pad_oov_mask, dim=1)

        summed_queries = torch.max(cosine_matrix_masked, dim=1) #/ query_lengths.unsqueeze(-1)

        summed_queries = summed_queries[0].unsqueeze(-1).transpose(1, 2)

        pos_res = self.pos_awareness(summed_queries)

        pos_res = pos_res.transpose(1, 2)

        pos_res = pos_res / 4

        pos_res = pos_res.unsqueeze(-1)

        pos_res_kernel_results = torch.exp(- torch.pow(pos_res - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))

        pos_res_masked = pos_res_kernel_results.squeeze() * document_pad_oov_mask.unsqueeze(-1)

        pos_res_per_kernel = torch.sum(pos_res_masked, 1) * self.nn_scaler

        pos_linear_result = self.dense_pos_aware(pos_res_per_kernel)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #
        # mean kernels
        #
        #kernel_results_masked2 = kernel_results_masked.clone()

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        #kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        #per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)

        per_kernel_query_mean = per_kernel_query / doc_lengths.view(-1,1,1)

        log_per_kernel_query_mean = torch.log(torch.clamp(per_kernel_query_mean, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 

        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        dense_mean_out = self.dense_mean(per_kernel_mean)
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out,pos_linear_result],dim=1))
        score = torch.squeeze(dense_comb_out,1) #torch.tanh(dense_out), 1)
        return score

    def get_param_stats(self):
        return "MM_light: dense w: "+str(self.dense.weight.data)+" b: "+str(self.dense.bias.data) +\
        "dense_mean weight: "+str(self.dense_mean.weight.data)+"b: "+str(self.dense_mean.bias.data) +\
        "dense_pos weight: "+str(self.dense_pos_aware.weight.data)+"b: "+str(self.dense_pos_aware.bias.data) +\
        "dense_comb weight: "+str(self.dense_comb.weight.data) + "scaler: "+str(self.nn_scaler.data)

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

        l_sigma += [bin_size] * (n_kernels - 1)
        return l_sigma

        
class Matchmaker_light_v5b(nn.Module):
    '''
    Paper: ...

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Matchmaker_light_v5b(n_kernels = config["mm_light_kernels"])

    def __init__(self,
                 n_kernels: int):

        super(Matchmaker_light_v5b, self).__init__()

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        self.pos_aware_convs = []
        self.pos_aware_linear = []
        self.pos_aware_size = []
        for i in range(2, 16):
            pos_aware_conv = nn.Conv1d(kernel_size=i,in_channels=1,out_channels=1,bias=False)
            torch.nn.init.ones_(pos_aware_conv.weight)
            #pos_aware_conv.weight.requires_grad = False
            self.pos_aware_convs.append(
                nn.Sequential(
                    nn.ConstantPad1d((0,i - 1), 0),
                    pos_aware_conv)#,
                    #nn.ReLU()) 
            )
            self.pos_aware_linear.append(nn.Linear(n_kernels, 1, bias=True))
            self.pos_aware_size.append(i)

        self.pos_aware_convs = nn.ModuleList(self.pos_aware_convs) # register conv as part of the model
        self.pos_aware_linear = nn.ModuleList(self.pos_aware_linear) # register linears as part of the model

        self.pos_aware_combine = nn.Linear(len(self.pos_aware_linear), 1, bias=True)


        #pos_awareness = nn.Conv1d(kernel_size=3,in_channels=1,out_channels=1,bias=False)
        #torch.nn.init.ones_(pos_awareness.weight)
        #pos_awareness.weight.requires_grad = False

        #self.pos_awareness = nn.Sequential(
        #            nn.ConstantPad1d((0,3 - 1), 0),
        #            pos_awareness)

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=True)
        self.dense_mean = nn.Linear(n_kernels, 1, bias=True)

        self.dense_comb = nn.Linear(3, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        torch.nn.init.uniform_(self.dense_mean.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------

        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # positional awareness
        #

        max_per_doc = torch.max(cosine_matrix_masked, dim=1)
        max_per_doc = max_per_doc[0].unsqueeze(-1).transpose(1, 2)

        #pos_aware_results = []
#
        #for i in range(len(self.pos_aware_convs)):
        #    conv_res = self.pos_aware_convs[i](max_per_doc)
        #    conv_res = (conv_res.transpose(1,2) / self.pos_aware_size[i]).unsqueeze(-1)
        #    kernel_res = torch.exp(- torch.pow(conv_res - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        #    
        #    kernel_res_masked = kernel_res.squeeze() * document_pad_oov_mask.unsqueeze(-1)
        #    kernel_res_summed = torch.sum(kernel_res_masked, 1) * self.nn_scaler
#
        #    pos_aware_results.append(self.pos_aware_linear[i](kernel_res_summed))
#
        #pos_aware_results = torch.cat(pos_aware_results,dim=1)
       #
        #pos_linear_result = self.pos_aware_combine(pos_aware_results)

        #
        # faster ? maybe with rolling window + sum (without conv) https://diegslva.github.io/2017-05-02-first-post/
        # (measure first - if conv trainable imporves perf! and what costs so much time)

        conv_res_tensor = torch.empty((cosine_matrix_extradim.shape[0],len(self.pos_aware_convs),cosine_matrix_extradim.shape[2],1),device=cosine_matrix.device)
        for i in range(len(self.pos_aware_convs)):
            conv_res = self.pos_aware_convs[i](max_per_doc)
            conv_res = (conv_res.transpose(1,2) / self.pos_aware_size[i])#.unsqueeze(-1)
            conv_res_tensor[:,i] = conv_res

        kernel_res2 = torch.exp(- torch.pow(conv_res_tensor - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_res_masked2 = kernel_res2.squeeze() * document_pad_oov_mask.view(document_pad_oov_mask.shape[0],1,-1,1)
        kernel_res_summed2 = torch.sum(kernel_res_masked2, dim=2) * self.nn_scaler
        
        pos_aware_linear_result = torch.empty((cosine_matrix_extradim.shape[0],len(self.pos_aware_linear)),device=cosine_matrix.device)
        for i in range(len(self.pos_aware_linear)):
            pos_aware_linear_result[:,i] = self.pos_aware_linear[i](kernel_res_summed2[:,i]).squeeze(-1)

        pos_linear_result = self.pos_aware_combine(pos_aware_linear_result)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #
        # mean kernels
        #
        #kernel_results_masked2 = kernel_results_masked.clone()

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        #kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        #per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)

        per_kernel_query_mean = per_kernel_query / doc_lengths.view(-1,1,1)

        log_per_kernel_query_mean = torch.log(torch.clamp(per_kernel_query_mean, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 

        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        dense_mean_out = self.dense_mean(per_kernel_mean)
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out,pos_linear_result],dim=1))
        score = torch.squeeze(dense_comb_out,1) #torch.tanh(dense_out), 1)
        return score

    def get_param_stats(self):
        return "MM_light: dense w: "+str(self.dense.weight.data)+" b: "+str(self.dense.bias.data) +\
        "dense_mean weight: "+str(self.dense_mean.weight.data)+"b: "+str(self.dense_mean.bias.data) +\
        "pos_aware_combine weight: "+str(self.pos_aware_combine.weight.data)+"b: "+str(self.pos_aware_combine.bias.data) +\
        "dense_comb weight: "+str(self.dense_comb.weight.data) + "scaler: "+str(self.nn_scaler.data)

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

        l_sigma += [bin_size] * (n_kernels - 1)
        return l_sigma

class Matchmaker_light_v5c(nn.Module):
    '''
    Paper: ...

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Matchmaker_light_v5c(n_kernels = config["mm_light_kernels"])

    def __init__(self,
                 n_kernels: int):

        super(Matchmaker_light_v5c, self).__init__()

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))

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
            self.pos_aware_linear.append(nn.Linear(n_kernels, 1, bias=True))
            self.pos_aware_size.append(i)

        #self.pos_aware_convs = nn.ModuleList(self.pos_aware_convs) # register conv as part of the model
        self.pos_aware_linear = nn.ModuleList(self.pos_aware_linear) # register linears as part of the model

        self.pos_aware_combine = nn.Linear(len(self.pos_aware_linear), 1, bias=True)


        #pos_awareness = nn.Conv1d(kernel_size=3,in_channels=1,out_channels=1,bias=False)
        #torch.nn.init.ones_(pos_awareness.weight)
        #pos_awareness.weight.requires_grad = False

        #self.pos_awareness = nn.Sequential(
        #            nn.ConstantPad1d((0,3 - 1), 0),
        #            pos_awareness)

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=True)
        self.dense_mean = nn.Linear(n_kernels, 1, bias=True)

        self.dense_comb = nn.Linear(3, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        torch.nn.init.uniform_(self.dense_mean.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------

        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # positional awareness
        #

        max_per_doc = torch.max(cosine_matrix_masked, dim=1)
        #max_per_doc = max_per_doc[0].unsqueeze(-1).transpose(1, 2)

        #pos_aware_results = []
#
        #for i in range(len(self.pos_aware_convs)):
        #    conv_res = self.pos_aware_convs[i](max_per_doc)
        #    conv_res = (conv_res.transpose(1,2) / self.pos_aware_size[i]).unsqueeze(-1)
        #    kernel_res = torch.exp(- torch.pow(conv_res - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        #    
        #    kernel_res_masked = kernel_res.squeeze() * document_pad_oov_mask.unsqueeze(-1)
        #    kernel_res_summed = torch.sum(kernel_res_masked, 1) * self.nn_scaler
#
        #    pos_aware_results.append(self.pos_aware_linear[i](kernel_res_summed))
#
        #pos_aware_results = torch.cat(pos_aware_results,dim=1)
       #
        #pos_linear_result = self.pos_aware_combine(pos_aware_results)

        #
        # faster ? maybe with rolling window + sum (without conv) https://diegslva.github.io/2017-05-02-first-post/
        # (measure first - if conv trainable imporves perf! and what costs so much time)

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
        
        pos_aware_linear_result = torch.empty((cosine_matrix_extradim.shape[0],len(self.pos_aware_linear)),device=cosine_matrix.device)
        for i in range(len(self.pos_aware_linear)):
            pos_aware_linear_result[:,i] = self.pos_aware_linear[i](kernel_res_summed2[:,i]).squeeze(-1)

        pos_linear_result = self.pos_aware_combine(pos_aware_linear_result)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #
        # mean kernels
        #
        #kernel_results_masked2 = kernel_results_masked.clone()

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        #kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        #per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)

        per_kernel_query_mean = per_kernel_query / doc_lengths.view(-1,1,1)

        log_per_kernel_query_mean = torch.log(torch.clamp(per_kernel_query_mean, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 

        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        dense_mean_out = self.dense_mean(per_kernel_mean)
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out,pos_linear_result],dim=1))
        score = torch.squeeze(dense_comb_out,1) #torch.tanh(dense_out), 1)
        return score

    def get_param_stats(self):
        return "MM_light: dense w: "+str(self.dense.weight.data)+" b: "+str(self.dense.bias.data) +\
        "dense_mean weight: "+str(self.dense_mean.weight.data)+"b: "+str(self.dense_mean.bias.data) +\
        "pos_aware_combine weight: "+str(self.pos_aware_combine.weight.data)+"b: "+str(self.pos_aware_combine.bias.data) +\
        "dense_comb weight: "+str(self.dense_comb.weight.data) + "scaler: "+str(self.nn_scaler.data)

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

        l_sigma += [bin_size] * (n_kernels - 1)
        return l_sigma


class Matchmaker_light_v5d(nn.Module):
    '''
    Paper: ...

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Matchmaker_light_v5d(word_embsize=word_embeddings_out_dim,n_kernels = config["mm_light_kernels"])

    def __init__(self,
                 word_embsize:int,
                 n_kernels: int):

        super(Matchmaker_light_v5d, self).__init__()

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()
        self.salc_W1 = nn.Linear(300, 1, bias=True)
        self.salc_W2 = nn.Linear(word_embsize, 300, bias=True)

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
            self.pos_aware_linear.append(nn.Linear(n_kernels, 1, bias=True))
            self.pos_aware_size.append(i)

        #self.pos_aware_convs = nn.ModuleList(self.pos_aware_convs) # register conv as part of the model
        self.pos_aware_linear = nn.ModuleList(self.pos_aware_linear) # register linears as part of the model
        self.pos_aware_div = nn.Parameter(torch.Tensor(self.pos_aware_size),requires_grad=True)
        self.pos_aware_combine = nn.Linear(len(self.pos_aware_linear), 1, bias=True)


        #pos_awareness = nn.Conv1d(kernel_size=3,in_channels=1,out_channels=1,bias=False)
        #torch.nn.init.ones_(pos_awareness.weight)
        #pos_awareness.weight.requires_grad = False

        #self.pos_awareness = nn.Sequential(
        #            nn.ConstantPad1d((0,3 - 1), 0),
        #            pos_awareness)

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=True)
        self.dense_mean = nn.Linear(n_kernels, 1, bias=True)

        self.dense_comb = nn.Linear(3, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        torch.nn.init.uniform_(self.dense_mean.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------

        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        #cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        qry_salc = self.salc_W1(torch.tanh(self.salc_W2(query_embeddings)))
        doc_salc = self.salc_W1(torch.tanh(self.salc_W2(document_embeddings)))
        
        qry_salc_matrix = qry_salc.repeat(1, 1, doc_salc.size()[1])
        doc_salc_matrix = doc_salc.repeat(1, 1, qry_salc.size()[1]).transpose(2,1)
        
        salc_matrix = torch.sigmoid(qry_salc_matrix) * torch.sigmoid(doc_salc_matrix)
        
        # match matrix
        match_matrix_masked = torch.tanh(cosine_matrix_masked * salc_matrix)
        match_matrix_extradim = match_matrix_masked.unsqueeze(-1)


        #
        # positional awareness
        #

        max_per_doc = torch.max(match_matrix_masked, dim=1)
        #max_per_doc = max_per_doc[0].unsqueeze(-1).transpose(1, 2)

        #pos_aware_results = []
#
        #for i in range(len(self.pos_aware_convs)):
        #    conv_res = self.pos_aware_convs[i](max_per_doc)
        #    conv_res = (conv_res.transpose(1,2) / self.pos_aware_size[i]).unsqueeze(-1)
        #    kernel_res = torch.exp(- torch.pow(conv_res - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        #    
        #    kernel_res_masked = kernel_res.squeeze() * document_pad_oov_mask.unsqueeze(-1)
        #    kernel_res_summed = torch.sum(kernel_res_masked, 1) * self.nn_scaler
#
        #    pos_aware_results.append(self.pos_aware_linear[i](kernel_res_summed))
#
        #pos_aware_results = torch.cat(pos_aware_results,dim=1)
       #
        #pos_linear_result = self.pos_aware_combine(pos_aware_results)

        #
        # faster ? maybe with rolling window + sum (without conv) https://diegslva.github.io/2017-05-02-first-post/
        # (measure first - if conv trainable imporves perf! and what costs so much time)

        conv_res_tensor = torch.empty((match_matrix_extradim.shape[0],len(self.pos_aware_size),match_matrix_extradim.shape[2],1),device=cosine_matrix.device)
        for i,prox_window in enumerate(self.pos_aware_size):

            sliding_window_i = max_per_doc[0].unfold(dimension=1, size=prox_window, step=1)
            sliding_sums = torch.sum(sliding_window_i,dim=2) / self.pos_aware_div[i]
            sliding_sums_padded = torch.nn.functional.pad(sliding_sums,(0,prox_window-1))

            #conv_res = self.pos_aware_convs[i](max_per_doc)
            #conv_res = (conv_res.transpose(1,2) / self.pos_aware_size[i])#.unsqueeze(-1)
            conv_res_tensor[:,i] = sliding_sums_padded.unsqueeze(-1) #.view(sliding_sums_padded.shape[0],1,-1,1)

        kernel_res2 = torch.exp(- torch.pow(conv_res_tensor - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_res_masked2 = kernel_res2.squeeze() * document_pad_oov_mask.view(document_pad_oov_mask.shape[0],1,-1,1)
        kernel_res_summed2 = torch.sum(kernel_res_masked2, dim=2) * self.nn_scaler
        
        pos_aware_linear_result = torch.empty((match_matrix_extradim.shape[0],len(self.pos_aware_linear)),device=cosine_matrix.device)
        for i in range(len(self.pos_aware_linear)):
            pos_aware_linear_result[:,i] = self.pos_aware_linear[i](kernel_res_summed2[:,i]).squeeze(-1)

        pos_linear_result = self.pos_aware_combine(pos_aware_linear_result)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(match_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #
        # mean kernels
        #
        #kernel_results_masked2 = kernel_results_masked.clone()

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        #kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        #per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)

        per_kernel_query_mean = per_kernel_query / doc_lengths.view(-1,1,1)

        log_per_kernel_query_mean = torch.log(torch.clamp(per_kernel_query_mean, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 

        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        dense_mean_out = self.dense_mean(per_kernel_mean)
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out,pos_linear_result],dim=1))
        score = torch.squeeze(dense_comb_out,1) #torch.tanh(dense_out), 1)
        return score

    def get_param_stats(self):
        return "MM_light: dense w: "+str(self.dense.weight.data)+" b: "+str(self.dense.bias.data) +\
        "dense_mean weight: "+str(self.dense_mean.weight.data)+"b: "+str(self.dense_mean.bias.data) +\
        "pos_aware_combine weight: "+str(self.pos_aware_combine.weight.data)+"b: "+str(self.pos_aware_combine.bias.data) +\
        "dense_comb weight: "+str(self.dense_comb.weight.data) +"pos_aware_div: "+str(self.pos_aware_div.data) + "scaler: "+str(self.nn_scaler.data)

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

        l_sigma += [bin_size] * (n_kernels - 1)
        return l_sigma


class Matchmaker_light_v5e(nn.Module):
    '''
    Paper: ...
    -> support bigger step size 
    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Matchmaker_light_v5e(word_embsize=word_embeddings_out_dim,n_kernels = config["mm_light_kernels"])

    def __init__(self,
                 word_embsize:int,
                 n_kernels: int):

        super(Matchmaker_light_v5e, self).__init__()

        # static - kernel size & magnitude variables

        self.mu2 = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, n_kernels)
        self.sigma2 = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, n_kernels)


        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()
        self.salc_W1 = nn.Linear(300, 1, bias=True)
        self.salc_W2 = nn.Linear(word_embsize, 300, bias=True)

        #self.pos_aware_convs = []
        self.pos_aware_linear = []
        self.pos_aware_size = []
        for i in range(3, 16, 3):#[2,5,10,15,20,30,40,50]:#range(2, 50, 5):
            #pos_aware_conv = nn.Conv1d(kernel_size=i,in_channels=1,out_channels=1,bias=False)
            #torch.nn.init.ones_(pos_aware_conv.weight)
            #pos_aware_conv.weight.requires_grad = False
            #self.pos_aware_convs.append(
            #    nn.Sequential(
            #        nn.ConstantPad1d((0,i - 1), 0),
            #        pos_aware_conv)#,
                    #nn.ReLU()) 
            #)
            self.pos_aware_linear.append(nn.Linear(n_kernels, 1, bias=True))
            self.pos_aware_size.append(i)

        #self.pos_aware_convs = nn.ModuleList(self.pos_aware_convs) # register conv as part of the model
        self.pos_aware_linear = nn.ModuleList(self.pos_aware_linear) # register linears as part of the model

        self.pos_aware_combine = nn.Linear(len(self.pos_aware_linear), 1, bias=True)


        #pos_awareness = nn.Conv1d(kernel_size=3,in_channels=1,out_channels=1,bias=False)
        #torch.nn.init.ones_(pos_awareness.weight)
        #pos_awareness.weight.requires_grad = False

        #self.pos_awareness = nn.Sequential(
        #            nn.ConstantPad1d((0,3 - 1), 0),
        #            pos_awareness)

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=True)
        self.dense_mean = nn.Linear(n_kernels, 1, bias=True)

        self.dense_comb = nn.Linear(3, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        torch.nn.init.uniform_(self.dense_mean.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------

        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        #cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        qry_salc = self.salc_W1(torch.tanh(self.salc_W2(query_embeddings)))
        doc_salc = self.salc_W1(torch.tanh(self.salc_W2(document_embeddings)))
        
        qry_salc_matrix = qry_salc.repeat(1, 1, doc_salc.size()[1])
        doc_salc_matrix = doc_salc.repeat(1, 1, qry_salc.size()[1]).transpose(2,1)
        
        salc_matrix = torch.sigmoid(qry_salc_matrix) * torch.sigmoid(doc_salc_matrix)
        
        # match matrix
        match_matrix_masked = torch.tanh(cosine_matrix_masked * salc_matrix)
        match_matrix_extradim = match_matrix_masked.unsqueeze(-1)


        #
        # positional awareness
        #

        max_per_doc = torch.max(match_matrix_masked, dim=1)
        #max_per_doc = max_per_doc[0].unsqueeze(-1).transpose(1, 2)

        #pos_aware_results = []
#
        #for i in range(len(self.pos_aware_convs)):
        #    conv_res = self.pos_aware_convs[i](max_per_doc)
        #    conv_res = (conv_res.transpose(1,2) / self.pos_aware_size[i]).unsqueeze(-1)
        #    kernel_res = torch.exp(- torch.pow(conv_res - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        #    
        #    kernel_res_masked = kernel_res.squeeze() * document_pad_oov_mask.unsqueeze(-1)
        #    kernel_res_summed = torch.sum(kernel_res_masked, 1) * self.nn_scaler
#
        #    pos_aware_results.append(self.pos_aware_linear[i](kernel_res_summed))
#
        #pos_aware_results = torch.cat(pos_aware_results,dim=1)
       #
        #pos_linear_result = self.pos_aware_combine(pos_aware_results)

        #
        # faster ? maybe with rolling window + sum (without conv) https://diegslva.github.io/2017-05-02-first-post/
        # (measure first - if conv trainable imporves perf! and what costs so much time)

        #conv_res_tensor = torch.empty((match_matrix_extradim.shape[0],len(self.pos_aware_size),match_matrix_extradim.shape[2],1),device=cosine_matrix.device)
        kernel_res_summed2 = torch.empty((match_matrix_extradim.shape[0],len(self.pos_aware_size),self.mu.shape[-1]),device=cosine_matrix.device)
        
        #pos_aware_results = []
        for i,prox_window in enumerate(self.pos_aware_size):

            sliding_window_i = max_per_doc[0].unfold(dimension=1, size=prox_window, step=int(prox_window/3))
            sliding_sums = torch.sum(sliding_window_i,dim=2) / prox_window
            #sliding_sums_padded = torch.nn.functional.pad(sliding_sums,(0,prox_window-1))

            #conv_res = self.pos_aware_convs[i](max_per_doc)
            #conv_res = (conv_res.transpose(1,2) / self.pos_aware_size[i])#.unsqueeze(-1)
            conv_res = sliding_sums.unsqueeze(-1) #.view(sliding_sums.shape[0],-1,1,1)

            kernel_res2 = torch.exp(- torch.pow(conv_res - self.mu2, 2) / (2 * torch.pow(self.sigma2, 2)))
        #kernel_res_masked2 = kernel_res2.squeeze() * document_pad_oov_mask.view(document_pad_oov_mask.shape[0],1,-1,1)
            kernel_res_summed2[:,i] = torch.sum(kernel_res2, dim=1) * self.nn_scaler
            #pos_aware_results.append(kernel_res_summed2)
        
        pos_aware_linear_result = torch.empty((match_matrix_extradim.shape[0],len(self.pos_aware_linear)),device=cosine_matrix.device)
        for i in range(len(self.pos_aware_linear)):
            pos_aware_linear_result[:,i] = self.pos_aware_linear[i](kernel_res_summed2[:,i]).squeeze(-1)

        pos_linear_result = self.pos_aware_combine(pos_aware_linear_result)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(match_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #
        # mean kernels
        #
        #kernel_results_masked2 = kernel_results_masked.clone()

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        #kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        #per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)

        per_kernel_query_mean = per_kernel_query / doc_lengths.view(-1,1,1)

        log_per_kernel_query_mean = torch.log(torch.clamp(per_kernel_query_mean, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 

        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        dense_mean_out = self.dense_mean(per_kernel_mean)
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out,pos_linear_result],dim=1))
        score = torch.squeeze(dense_comb_out,1) #torch.tanh(dense_out), 1)
        return score

    def get_param_stats(self):
        return "MM_light: dense w: "+str(self.dense.weight.data)+" b: "+str(self.dense.bias.data) +\
        "dense_mean weight: "+str(self.dense_mean.weight.data)+"b: "+str(self.dense_mean.bias.data) +\
        "pos_aware_combine weight: "+str(self.pos_aware_combine.weight.data)+"b: "+str(self.pos_aware_combine.bias.data) +\
        "dense_comb weight: "+str(self.dense_comb.weight.data) + "scaler: "+str(self.nn_scaler.data)

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

        l_sigma += [bin_size] * (n_kernels - 1)
        return l_sigma



class Matchmaker_light_v6a(nn.Module):
    '''
    Paper: ...

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim,vocabulary_size):
        return Matchmaker_light_v6a(n_kernels = config["mm_light_kernels"],vocabulary_size=vocabulary_size)

    def __init__(self,
                 n_kernels: int,
                 vocabulary_size:int):

        super(Matchmaker_light_v6a, self).__init__()

        self.salience_weights = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=1, padding_idx=0)
        torch.nn.init.constant_(self.salience_weights.weight,1)

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.match_matrix = CosineMatrixAttention()

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
            self.pos_aware_linear.append(nn.Linear(n_kernels, 1, bias=True))
            self.pos_aware_size.append(i)

        #self.pos_aware_convs = nn.ModuleList(self.pos_aware_convs) # register conv as part of the model
        self.pos_aware_linear = nn.ModuleList(self.pos_aware_linear) # register linears as part of the model

        self.pos_aware_combine = nn.Linear(len(self.pos_aware_linear), 1, bias=True)


        #pos_awareness = nn.Conv1d(kernel_size=3,in_channels=1,out_channels=1,bias=False)
        #torch.nn.init.ones_(pos_awareness.weight)
        #pos_awareness.weight.requires_grad = False

        #self.pos_awareness = nn.Sequential(
        #            nn.ConstantPad1d((0,3 - 1), 0),
        #            pos_awareness)

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=True)
        self.dense_mean = nn.Linear(n_kernels, 1, bias=True)

        self.dense_comb = nn.Linear(3, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        torch.nn.init.uniform_(self.dense_mean.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor,
                query_ids: torch.Tensor, document_ids: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_salience = torch.sigmoid(self.salience_weights(query_ids))
        document_salience = torch.sigmoid(self.salience_weights(document_ids))

        query_by_doc_salience = torch.bmm(query_salience, document_salience.transpose(-1, -2))
        #query_by_doc_salience_view = query_by_doc_salience.unsqueeze(-1)

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------

        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.match_matrix.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = torch.tanh(cosine_matrix * query_by_doc_salience) * query_by_doc_mask
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # positional awareness
        #

        max_per_doc = torch.max(cosine_matrix_masked, dim=1)
        #max_per_doc = max_per_doc[0].unsqueeze(-1).transpose(1, 2)

        #pos_aware_results = []
#
        #for i in range(len(self.pos_aware_convs)):
        #    conv_res = self.pos_aware_convs[i](max_per_doc)
        #    conv_res = (conv_res.transpose(1,2) / self.pos_aware_size[i]).unsqueeze(-1)
        #    kernel_res = torch.exp(- torch.pow(conv_res - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        #    
        #    kernel_res_masked = kernel_res.squeeze() * document_pad_oov_mask.unsqueeze(-1)
        #    kernel_res_summed = torch.sum(kernel_res_masked, 1) * self.nn_scaler
#
        #    pos_aware_results.append(self.pos_aware_linear[i](kernel_res_summed))
#
        #pos_aware_results = torch.cat(pos_aware_results,dim=1)
       #
        #pos_linear_result = self.pos_aware_combine(pos_aware_results)

        #
        # faster ? maybe with rolling window + sum (without conv) https://diegslva.github.io/2017-05-02-first-post/
        # (measure first - if conv trainable imporves perf! and what costs so much time)

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
        
        pos_aware_linear_result = torch.empty((cosine_matrix_extradim.shape[0],len(self.pos_aware_linear)),device=cosine_matrix.device)
        for i in range(len(self.pos_aware_linear)):
            pos_aware_linear_result[:,i] = self.pos_aware_linear[i](kernel_res_summed2[:,i]).squeeze(-1)

        pos_linear_result = self.pos_aware_combine(pos_aware_linear_result)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #
        # mean kernels
        #
        #kernel_results_masked2 = kernel_results_masked.clone()

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        #kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        #per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)

        per_kernel_query_mean = per_kernel_query / doc_lengths.view(-1,1,1)

        log_per_kernel_query_mean = torch.log(torch.clamp(per_kernel_query_mean, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 

        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        dense_mean_out = self.dense_mean(per_kernel_mean)
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out,pos_linear_result],dim=1))
        score = torch.squeeze(dense_comb_out,1) #torch.tanh(dense_out), 1)
        return score

    def get_param_stats(self):
        return "MM_light: dense w: "+str(self.dense.weight.data)+" b: "+str(self.dense.bias.data) +\
        "dense_mean weight: "+str(self.dense_mean.weight.data)+"b: "+str(self.dense_mean.bias.data) +\
        "pos_aware_combine weight: "+str(self.pos_aware_combine.weight.data)+"b: "+str(self.pos_aware_combine.bias.data) +\
        "dense_comb weight: "+str(self.dense_comb.weight.data) + "scaler: "+str(self.nn_scaler.data)

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

        l_sigma += [bin_size] * (n_kernels - 1)
        return l_sigma


class Matchmaker_light_v7(nn.Module):
    '''
    Paper: ...

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim,vocabulary_size):
        return Matchmaker_light_v7(config["mm_light_kernels"],word_embeddings_out_dim,vocabulary_size=vocabulary_size)

    def __init__(self,
                 n_kernels: int,
                 word_embeddings_out_dim,
                 vocabulary_size:int):

        super(Matchmaker_light_v7, self).__init__()

        #self.salience_weights = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=1, padding_idx=0)
        #torch.nn.init.constant_(self.salience_weights.weight,1)

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))

        self.stacked_att = StackedSelfAttentionEncoder(input_dim=word_embeddings_out_dim,
                                                       hidden_dim=word_embeddings_out_dim,
                                                       projection_dim=32,
                                                       feedforward_hidden_dim=100,
                                                       num_layers=1,
                                                       num_attention_heads=32,
                                                       )



        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.match_matrix = CosineMatrixAttention()

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
            self.pos_aware_linear.append(nn.Linear(n_kernels, 1, bias=True))
            self.pos_aware_size.append(i)

        #self.pos_aware_convs = nn.ModuleList(self.pos_aware_convs) # register conv as part of the model
        self.pos_aware_linear = nn.ModuleList(self.pos_aware_linear) # register linears as part of the model

        self.pos_aware_combine = nn.Linear(len(self.pos_aware_linear), 1, bias=True)


        #pos_awareness = nn.Conv1d(kernel_size=3,in_channels=1,out_channels=1,bias=False)
        #torch.nn.init.ones_(pos_awareness.weight)
        #pos_awareness.weight.requires_grad = False

        #self.pos_awareness = nn.Sequential(
        #            nn.ConstantPad1d((0,3 - 1), 0),
        #            pos_awareness)

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=True)
        self.dense_mean = nn.Linear(n_kernels, 1, bias=True)

        self.dense_comb = nn.Linear(3, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        torch.nn.init.uniform_(self.dense_mean.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor,
                query_ids: torch.Tensor, document_ids: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_embeddings_context = self.stacked_att(query_embeddings,query_pad_oov_mask)
        document_embeddings_context = self.stacked_att(document_embeddings,document_pad_oov_mask)

        query_embeddings = torch.cat([query_embeddings,query_embeddings_context],dim=2)
        document_embeddings = torch.cat([document_embeddings,document_embeddings_context],dim=2)



        #query_salience = torch.sigmoid(self.salience_weights(query_ids))
        #document_salience = torch.sigmoid(self.salience_weights(document_ids))

        #query_by_doc_salience = torch.bmm(query_salience, document_salience.transpose(-1, -2))
        #query_by_doc_salience_view = query_by_doc_salience.unsqueeze(-1)

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------

        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.match_matrix.forward(query_embeddings, document_embeddings)
        cosine_matrix_masked = torch.tanh(cosine_matrix) * query_by_doc_mask
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # positional awareness
        #

        max_per_doc = torch.max(cosine_matrix_masked, dim=1)

        #
        # faster ? maybe with rolling window + sum (without conv) https://diegslva.github.io/2017-05-02-first-post/
        # (measure first - if conv trainable imporves perf! and what costs so much time)

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
        
        pos_aware_linear_result = torch.empty((cosine_matrix_extradim.shape[0],len(self.pos_aware_linear)),device=cosine_matrix.device)
        for i in range(len(self.pos_aware_linear)):
            pos_aware_linear_result[:,i] = self.pos_aware_linear[i](kernel_res_summed2[:,i]).squeeze(-1)

        pos_linear_result = self.pos_aware_combine(pos_aware_linear_result)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #
        # mean kernels
        #
        #kernel_results_masked2 = kernel_results_masked.clone()

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        #kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        #per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)

        per_kernel_query_mean = per_kernel_query / doc_lengths.view(-1,1,1)

        log_per_kernel_query_mean = torch.log(torch.clamp(per_kernel_query_mean, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 

        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        dense_mean_out = self.dense_mean(per_kernel_mean)
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out,pos_linear_result],dim=1))
        score = torch.squeeze(dense_comb_out,1) #torch.tanh(dense_out), 1)
        return score

    def get_param_stats(self):
        return "MM_light: dense w: "+str(self.dense.weight.data)+" b: "+str(self.dense.bias.data) +\
        "dense_mean weight: "+str(self.dense_mean.weight.data)+"b: "+str(self.dense_mean.bias.data) +\
        "pos_aware_combine weight: "+str(self.pos_aware_combine.weight.data)+"b: "+str(self.pos_aware_combine.bias.data) +\
        "dense_comb weight: "+str(self.dense_comb.weight.data) + "scaler: "+str(self.nn_scaler.data)

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

        l_sigma += [bin_size] * (n_kernels - 1)
        return l_sigma