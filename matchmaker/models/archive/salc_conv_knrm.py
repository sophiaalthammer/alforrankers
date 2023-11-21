import pdb
from typing import Dict, Iterator, List
import pickle
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          

class Salc_Conv_KNRM(nn.Module):

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Salc_Conv_KNRM(n_kernels=config["salc_conv_knrm_kernels"],
                              conv_out_dim=config["salc_conv_knrm_conv_out_dim"],
                              dropi_rate=config["salc_conv_knrm_dropi"],
                              drops_rate=config["salc_conv_knrm_drops"],
                              word_embsize=word_embeddings_out_dim,
                              salc_dim=config["salc_conv_knrm_salc_dim"])

    def __init__(self,
                 n_kernels: int,
                 conv_out_dim: int,
                 dropi_rate: float,
                 drops_rate: float,
                 word_embsize:int,
                 salc_dim: int) -> None:
        super().__init__()

        self.dropi = nn.Dropout(dropi_rate)
        self.drops = nn.Dropout(drops_rate)
        
        self.mu = nn.Parameter(torch.tensor(self.kernel_mus(n_kernels), 
                               dtype=torch.float32, requires_grad=False).view(1, 1, 1, n_kernels),requires_grad=False)
        self.sigma = nn.Parameter(torch.tensor(self.kernel_sigmas(n_kernels),
                                  dtype=torch.float32, requires_grad=False).view(1, 1, 1, n_kernels),requires_grad=False)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))

        self.conv_1 = nn.Sequential(
            nn.Conv1d(kernel_size = 1, in_channels=word_embsize, out_channels=conv_out_dim),
            nn.ReLU())

        self.conv_2 = nn.Sequential(
            nn.Conv1d(kernel_size = 2, in_channels=word_embsize, out_channels=conv_out_dim),
            nn.ReLU())

        self.conv_3 = nn.Sequential(
            nn.Conv1d(kernel_size = 3, in_channels=word_embsize, out_channels=conv_out_dim),
            nn.ReLU())

        self.cosine_module = CosineMatrixAttention()

        #weights for term saliency
        self.salc_W1 = nn.Linear(salc_dim, 1, bias=True)
        self.salc_W2 = nn.Linear(conv_out_dim, salc_dim, bias=True)
        
        self.dense = nn.Linear(n_kernels * 9, 1, bias=False) 

        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        
    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor) -> torch.Tensor:

        #
        # prepare embedding tensors
        # -------------------------------------------------------
        
        document_pad_mask = document_pad_oov_mask
        query_pad_mask = query_pad_oov_mask

        query_by_doc_mask = torch.bmm(query_pad_mask.unsqueeze(-1), document_pad_mask.unsqueeze(-1).transpose(-1, -2))
        
        query_embeddings = self.dropi(query_embeddings)
        document_embeddings = self.dropi(document_embeddings)

        query_embeddings_t = query_embeddings.transpose(1, 2)
        query_embeddings_t_p2 = nn.functional.pad(query_embeddings_t,(0,1)) 
        query_embeddings_t_p3 = nn.functional.pad(query_embeddings_t,(0,2)) 

        document_embeddings_t = document_embeddings.transpose(1, 2) 
        document_embeddings_t_p2 = nn.functional.pad(document_embeddings_t,(0,1))
        document_embeddings_t_p3 = nn.functional.pad(document_embeddings_t,(0,2))

        query_conv_1 = self.conv_1(query_embeddings_t).transpose(1, 2) 
        query_conv_2 = self.conv_2(query_embeddings_t_p2).transpose(1, 2)
        query_conv_3 = self.conv_3(query_embeddings_t_p3).transpose(1, 2)

        document_conv_1 = self.conv_1(document_embeddings_t).transpose(1, 2) 
        document_conv_2 = self.conv_2(document_embeddings_t_p2).transpose(1, 2)
        document_conv_3 = self.conv_3(document_embeddings_t_p3).transpose(1, 2)

        #
        # similarity matrix & gaussian kernels & soft TF for all conv combinations
        # -------------------------------------------------------

        sim_q1_d1 = self.forward_matrix_kernel_pooling(query_conv_1, document_conv_1, query_by_doc_mask, query_pad_mask)
        sim_q1_d2 = self.forward_matrix_kernel_pooling(query_conv_1, document_conv_2, query_by_doc_mask, query_pad_mask)
        sim_q1_d3 = self.forward_matrix_kernel_pooling(query_conv_1, document_conv_3, query_by_doc_mask, query_pad_mask)
        
        sim_q2_d1 = self.forward_matrix_kernel_pooling(query_conv_2, document_conv_1, query_by_doc_mask, query_pad_mask)
        sim_q2_d2 = self.forward_matrix_kernel_pooling(query_conv_2, document_conv_2, query_by_doc_mask, query_pad_mask)
        sim_q2_d3 = self.forward_matrix_kernel_pooling(query_conv_2, document_conv_3, query_by_doc_mask, query_pad_mask)
        
        sim_q3_d1 = self.forward_matrix_kernel_pooling(query_conv_3, document_conv_1, query_by_doc_mask, query_pad_mask)
        sim_q3_d2 = self.forward_matrix_kernel_pooling(query_conv_3, document_conv_2, query_by_doc_mask, query_pad_mask)
        sim_q3_d3 = self.forward_matrix_kernel_pooling(query_conv_3, document_conv_3, query_by_doc_mask, query_pad_mask)
        
        # concatenating all grams to one vector 
        tfs = torch.cat([
            sim_q1_d1,sim_q1_d2,sim_q1_d3,
            sim_q2_d1,sim_q2_d2,sim_q2_d3,
            sim_q3_d1,sim_q3_d2,sim_q3_d3], 1)
        
        # scaling down softtf
        tfs = tfs * self.nn_scaler
        
        #
        # "Learning to rank" layer
        # -------------------------------------------------------


        dense_out = self.dense(tfs)

        output = torch.squeeze(dense_out, 1)
        return output
 
    ## create a match matrix between query & document terms
    def forward_matrix_kernel_pooling(self, query_tensor, document_tensor, query_by_doc_mask, query_pad_oov_mask):

        #
        # cosine matrix
        # -------------------------------------------------------
        # shape: (batch, query_max, doc_max)
        
        cosine_matrix = self.cosine_module.forward(query_tensor, document_tensor)
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        
        # saliency
        query_tensor_droped = self.drops(query_tensor)
        document_tensor_droped = self.drops(document_tensor)
        
        qry_salc = self.salc_W1(torch.tanh(self.salc_W2(query_tensor_droped)))
        doc_salc = self.salc_W1(torch.tanh(self.salc_W2(document_tensor_droped)))
        
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
        kernel_results_masked = raw_kernel_results * query_by_doc_mask.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) #0.01
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1)

        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        return per_kernel

    def get_param_stats(self):
        return "Salc_Conv_KNRM: dense w: "+str(self.dense.weight.data)+ "scaler: "+str(self.nn_scaler.data)

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