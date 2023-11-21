from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          


class Conv_KNRM_LN(Model):
    '''
    Paper: Convolutional Neural Networks for Soſt-Matching N-Grams in Ad-hoc Search, Dai et al. WSDM 18

    third-hand reference: https://github.com/NTMC-Community/MatchZoo/blob/master/matchzoo/models/conv_knrm.py (tensorflow)
    https://github.com/thunlp/EntityDuetNeuralRanking/blob/master/baselines/CKNRM.py (pytorch)

    '''

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 vocab: Vocabulary,
                 n_kernels: int,
                 conv_out_dim:int,
                 cuda_device: int) -> None:
        super().__init__(vocab)

        self.word_embeddings = word_embeddings

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)

        # stride is always 1, = the size of the steps for each sliding step
        self.conv_1 = nn.Sequential(
            nn.Conv1d(kernel_size = 1, in_channels=word_embeddings.get_output_dim(), out_channels=conv_out_dim),
            nn.ReLU())
#
        self.conv_2 = nn.Sequential(
            nn.Conv1d(kernel_size = 2, in_channels=word_embeddings.get_output_dim(), out_channels=conv_out_dim),
            nn.ReLU())
#
        self.conv_3 = nn.Sequential(
            nn.Conv1d(kernel_size = 3, in_channels=word_embeddings.get_output_dim(), out_channels=conv_out_dim),
            nn.ReLU())

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()
        #self.dropout = nn.Dropout(0.5)
        
        self.length_norm_factor = nn.Parameter(torch.full([n_kernels * 9],0.5,requires_grad=True).cuda())

        # *9 because we concat the 3x3 conv match sums together before the dense layer
        self.dense = nn.Linear(n_kernels * 9, 1, bias=False) 

        # init with small weights, otherwise the dense output is way to high fot
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

        #self.conv_uni = nn.Sequential(
        #    nn.Conv2d(1, 128, (1, word_embeddings.get_output_dim())),
        #    nn.ReLU()
        #)
#
        #self.conv_bi = nn.Sequential(
        #    nn.Conv2d(1, 128, (2, word_embeddings.get_output_dim())),
        #    nn.ReLU()
        #)
        #self.conv_tri = nn.Sequential(
        #    nn.Conv2d(1, 128, (3, word_embeddings.get_output_dim())),
        #    nn.ReLU()
        #)
        #self.dense_f = nn.Linear(n_kernels * 9, 1, 1)


    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor],
                query_length: torch.Tensor, document_length: torch.Tensor) -> torch.Tensor:

        #
        # prepare embedding tensors
        # -------------------------------------------------------

        #if query["tokens"].shape[1] == 1:
        #    query["tokens"] = nn.functional.pad(query["tokens"],(0, 3))
#
        #if query["tokens"].shape[1] == 2:
        #    query["tokens"] = nn.functional.pad(query["tokens"],(0, 2))
#
        #if query["tokens"].shape[1] == 3:
        #    query["tokens"] = nn.functional.pad(query["tokens"],(0, 1))


        # we assume 1 is the unknown token, 0 is padding - both need to be removed

        # shape: (batch, query_max)
        query_pad_oov_mask = (query["tokens"] > 1).float()
        # shape: (batch, doc_max)
        document_pad_oov_mask = (document["tokens"] > 1).float()

        # shape: (batch, query_max)
        query_pad_mask = (query["tokens"] > 0).float()
        # shape: (batch, doc_max)
        document_pad_mask = (document["tokens"] > 0).float()


        query_by_doc_mask = torch.bmm(query_pad_mask.unsqueeze(-1), document_pad_mask.unsqueeze(-1).transpose(-1, -2))
        #query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        # shape: (batch, query_max,emb_dim)
        query_embeddings = self.word_embeddings(query) * query_pad_oov_mask.unsqueeze(-1)
        # shape: (batch, document_max,emb_dim)
        document_embeddings = self.word_embeddings(document) * document_pad_oov_mask.unsqueeze(-1)

        #
        # 1-3 gram convolutions for query & document
        # -------------------------------------------------------

        #qwu_embed = torch.transpose(torch.squeeze(self.conv_uni(query_embeddings.view(query_embeddings.size()[0], 1, -1, 300)),3), -1, -2)
        #qwb_embed = torch.transpose(torch.squeeze(self.conv_bi (query_embeddings.view(query_embeddings.size()[0], 1, -1, 300)),3), -1, -2)
        #qwt_embed = torch.transpose(torch.squeeze(self.conv_tri(query_embeddings.view(query_embeddings.size()[0], 1, -1, 300)),3), -1, -2)
        #dwu_embed = torch.squeeze(self.conv_uni(document_embeddings.view(document_embeddings.size()[0], 1, -1, 300)),3)
        #dwb_embed = torch.squeeze(self.conv_bi (document_embeddings.view(document_embeddings.size()[0], 1, -1, 300)),3)
        #dwt_embed = torch.squeeze(self.conv_tri(document_embeddings.view(document_embeddings.size()[0], 1, -1, 300)),3)
#
        #qwu_embed_norm = F.normalize(qwu_embed, p=2, dim=2, eps=1e-10)
        #qwb_embed_norm = F.normalize(qwb_embed, p=2, dim=2, eps=1e-10)
        #qwt_embed_norm = F.normalize(qwt_embed, p=2, dim=2, eps=1e-10)
        #dwu_embed_norm = F.normalize(dwu_embed, p=2, dim=1, eps=1e-10)
        #dwb_embed_norm = F.normalize(dwb_embed, p=2, dim=1, eps=1e-10)
        #dwt_embed_norm = F.normalize(dwt_embed, p=2, dim=1, eps=1e-10)
        #mask_qw = query_pad_mask.view(query_pad_mask.size()[0], query_pad_mask.size()[1], 1)
        #mask_dw = document_pad_mask.view(document_pad_mask.size()[0], 1, document_pad_mask.size()[1], 1)
        #mask_qwu = mask_qw[:, :query_pad_mask.size()[1] - (1 - 1), :]
        #mask_qwb = mask_qw[:, :query_pad_mask.size()[1] - (2 - 1), :]
        #mask_qwt = mask_qw[:, :query_pad_mask.size()[1] - (3 - 1), :]
        #mask_dwu = mask_dw[:, :, :document_pad_mask.size()[1] - (1 - 1), :]
        #mask_dwb = mask_dw[:, :, :document_pad_mask.size()[1] - (2 - 1), :]
        #mask_dwt = mask_dw[:, :, :document_pad_mask.size()[1] - (3 - 1), :]
        #log_pooling_sum_wwuu = self.get_intersect_matrix(qwu_embed_norm, dwu_embed_norm, mask_qwu, mask_dwu)
        #log_pooling_sum_wwut = self.get_intersect_matrix(qwu_embed_norm, dwt_embed_norm, mask_qwu, mask_dwt)
        #log_pooling_sum_wwub = self.get_intersect_matrix(qwu_embed_norm, dwb_embed_norm, mask_qwu, mask_dwb)
        #log_pooling_sum_wwbu = self.get_intersect_matrix(qwb_embed_norm, dwu_embed_norm, mask_qwb, mask_dwu)
        #log_pooling_sum_wwtu = self.get_intersect_matrix(qwt_embed_norm, dwu_embed_norm, mask_qwt, mask_dwu)
#
        #log_pooling_sum_wwbb = self.get_intersect_matrix(qwb_embed_norm, dwb_embed_norm, mask_qwb, mask_dwb)
        #log_pooling_sum_wwbt = self.get_intersect_matrix(qwb_embed_norm, dwt_embed_norm, mask_qwb, mask_dwt)
        #log_pooling_sum_wwtb = self.get_intersect_matrix(qwt_embed_norm, dwb_embed_norm, mask_qwt, mask_dwb)
        #log_pooling_sum_wwtt = self.get_intersect_matrix(qwt_embed_norm, dwt_embed_norm, mask_qwt, mask_dwt)
        #log_pooling_sum = torch.cat([ log_pooling_sum_wwuu, log_pooling_sum_wwut, log_pooling_sum_wwub, log_pooling_sum_wwbu, log_pooling_sum_wwtu,\
        #    log_pooling_sum_wwbb, log_pooling_sum_wwbt, log_pooling_sum_wwtb, log_pooling_sum_wwtt], 1)
        #output = torch.squeeze(F.tanh(self.dense_f(log_pooling_sum)), 1)
        #return output

        # !! conv1d requires tensor in shape: [batch, emb_dim, sequence_length ]
        # so we transpose embedding tensors from : [batch, sequence_length,emb_dim] to [batch, emb_dim, sequence_length ]
        # feed that into the conv1d and reshape output from [batch, conv1d_out_channels, sequence_length ] 
        # to [batch, sequence_length, conv1d_out_channels]

        query_embeddings_t = query_embeddings.transpose(1, 2) # doesn't need padding because kernel_size = 1
        query_embeddings_t_p2 = nn.functional.pad(query_embeddings_t,(0,1)) # we add kernel_size - 1 padding, so that output has same size (and we don't ignore last column)
        query_embeddings_t_p3 = nn.functional.pad(query_embeddings_t,(0,2)) # we add kernel_size - 1 padding, so that output has same size (and we don't ignore last 2 columns)

        document_embeddings_t = document_embeddings.transpose(1, 2) # doesn't need padding because kernel_size = 1
        document_embeddings_t_p2 = nn.functional.pad(document_embeddings_t,(0,1)) # we add kernel_size - 1 padding, so that output has same size (and we don't ignore last column)
        document_embeddings_t_p3 = nn.functional.pad(document_embeddings_t,(0,2)) # we add kernel_size - 1 padding, so that output has same size (and we don't ignore last 2 columns)

        query_conv_1 = self.conv_1(query_embeddings_t).transpose(1, 2) 
        query_conv_2 = self.conv_2(query_embeddings_t_p2).transpose(1, 2)
        query_conv_3 = self.conv_3(query_embeddings_t_p3).transpose(1, 2)

        document_conv_1 = self.conv_1(document_embeddings_t).transpose(1, 2) 
        document_conv_2 = self.conv_2(document_embeddings_t_p2).transpose(1, 2)
        document_conv_3 = self.conv_3(document_embeddings_t_p3).transpose(1, 2)

        #
        # similarity matrix & gaussian kernels & soft TF for all conv combinations
        # -------------------------------------------------------

        # TODO question about the query_by_doc_mask - shouldn't we remove the last & 2-nd last "1" in every sample - row based on the conv, because l_out is < l_in so we leave the last element wrongly
        sim_q1_d1 = self.forward_matrix_kernel_pooling(query_conv_1, document_conv_1, query_by_doc_mask, query_pad_mask)
        sim_q1_d2 = self.forward_matrix_kernel_pooling(query_conv_1, document_conv_2, query_by_doc_mask, query_pad_mask)
        sim_q1_d3 = self.forward_matrix_kernel_pooling(query_conv_1, document_conv_3, query_by_doc_mask, query_pad_mask)

        sim_q2_d1 = self.forward_matrix_kernel_pooling(query_conv_2, document_conv_1, query_by_doc_mask, query_pad_mask)
        sim_q2_d2 = self.forward_matrix_kernel_pooling(query_conv_2, document_conv_2, query_by_doc_mask, query_pad_mask)
        sim_q2_d3 = self.forward_matrix_kernel_pooling(query_conv_2, document_conv_3, query_by_doc_mask, query_pad_mask)

        sim_q3_d1 = self.forward_matrix_kernel_pooling(query_conv_3, document_conv_1, query_by_doc_mask, query_pad_mask)
        sim_q3_d2 = self.forward_matrix_kernel_pooling(query_conv_3, document_conv_2, query_by_doc_mask, query_pad_mask)
        sim_q3_d3 = self.forward_matrix_kernel_pooling(query_conv_3, document_conv_3, query_by_doc_mask, query_pad_mask)

        #
        # "Learning to rank" layer
        # -------------------------------------------------------
        l_norm = (1 - self.length_norm_factor + self.length_norm_factor * document_length.float().unsqueeze(1)) # todo

        all_grams = torch.cat([
            sim_q1_d1,sim_q1_d2,sim_q1_d3,
            sim_q2_d1,sim_q2_d2,sim_q2_d3,
            sim_q3_d1,sim_q3_d2,sim_q3_d3],1)

        length_normalized = all_grams / l_norm

        dense_out = self.dense(length_normalized)
        tanh_out = torch.tanh(dense_out)

        output = torch.squeeze(tanh_out, 1)
        return output

    #def get_intersect_matrix(self, q_embed, d_embed, atten_q, atten_d):
#
    #    sim = torch.bmm(q_embed, d_embed).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[2], 1)
    #    pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2)) * atten_d
    #    pooling_sum = torch.sum(pooling_value, 2)
    #    log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * 0.01 * atten_q
    #    log_pooling_sum = torch.sum(log_pooling_sum, 1)
    #    return log_pooling_sum

    #
    # create a match matrix between query & document terms
    #
    def forward_matrix_kernel_pooling(self, query_tensor, document_tensor, query_by_doc_mask, query_pad_oov_mask):

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
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) * 0.01
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values

        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        return per_kernel

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
