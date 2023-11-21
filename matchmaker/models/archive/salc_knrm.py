from typing import Dict, Iterator, List
import pdb
import pickle
import numpy as np
from collections import defaultdict

from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          
    
class Salc_KNRM(nn.Module):

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Salc_KNRM(n_kernels=config["salc_knrm_kernels"],
                         dropi_rate=config["salc_knrm_dropi"],
                         drops_rate=config["salc_knrm_drops"],
                         word_embsize=word_embeddings_out_dim,
                         salc_dim=config["salc_knrm_salc_dim"])

    def __init__(self,
                 n_kernels: int,
                 dropi_rate: float,
                 drops_rate: float,
                 word_embsize:int,
                 salc_dim: int) -> None:
        
        super().__init__()
        
        self.dropi = nn.Dropout(dropi_rate)
        self.drops = nn.Dropout(drops_rate)

        # static - kernel size & magnitude tensors
        self.mu = nn.Parameter(torch.tensor(self.kernel_mus(n_kernels), 
                               dtype=torch.float32, requires_grad=False).view(1, 1, 1, n_kernels),requires_grad=False)
        self.sigma = nn.Parameter(torch.tensor(self.kernel_sigmas(n_kernels),
                                  dtype=torch.float32, requires_grad=False).view(1, 1, 1, n_kernels),requires_grad=False)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        #weights for term saliency
        self.salc_W1 = nn.Linear(salc_dim, 1, bias=True)
        self.salc_W2 = nn.Linear(word_embsize, salc_dim, bias=True)
        
        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo

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
        
        # saliency
        query_embeddings_droped = self.drops(query_embeddings)
        document_embeddings_droped = self.drops(document_embeddings)
        
        qry_salc = self.salc_W1(torch.tanh(self.salc_W2(query_embeddings_droped)))
        doc_salc = self.salc_W1(torch.tanh(self.salc_W2(document_embeddings_droped)))
        
        qry_salc_matrix = qry_salc.repeat(1, 1, doc_salc.size()[1])
        doc_salc_matrix = doc_salc.repeat(1, 1, qry_salc.size()[1]).transpose(2,1)
        
        salc_matrix = torch.sigmoid(qry_salc_matrix) * torch.sigmoid(doc_salc_matrix)
        
        # match matrix
        match_matrix_masked = torch.tanh(cosine_matrix_masked * salc_matrix)
        match_matrix_extradim = match_matrix_masked.unsqueeze(-1)


        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------
        raw_kernel_results = torch.exp(- torch.pow(match_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler #0.01
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) 

        tfs = torch.sum(log_per_kernel_query_masked, 1) 

        # scaling down softtf
        tfs = tfs * self.nn_scaler

        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(tfs)
        #tanh_out = torch.tanh(dense_out)

        score = torch.squeeze(dense_out, 1)
        return score

    def get_param_stats(self):
        return "Salc_KNRM: dense w: "+str(self.dense.weight.data)+ "scaler: "+str(self.nn_scaler.data)

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