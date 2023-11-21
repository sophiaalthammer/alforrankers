from typing import Dict, Iterator, List,Tuple
from collections import OrderedDict

import torch
import torch.nn as nn                              
import torch.nn.functional as F
from torch.autograd import Variable

from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          
from allennlp.modules.matrix_attention.dot_product_matrix_attention import *                          

from matchmaker.modules.masked_softmax import MaskedSoftmax
from allennlp.nn.util import masked_softmax

class Matchmaker_vNext_v1(nn.Module):
    '''
    
    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Matchmaker_vNext_v1(word_embeddings_out_dim)

    def __init__(self,
                 _embsize:int):
                 
        super(Matchmaker_vNext_v1,self).__init__()

        n_kernels = 11
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)

        self.cosine_module = CosineMatrixAttention()
        self.dot_module = DotProductMatrixAttention()
        self.masked_softmax = MaskedSoftmax()

        self.contextualization = StackedSelfAttentionEncoder(input_dim=_embsize,
                 hidden_dim=_embsize,
                 projection_dim=64,
                 feedforward_hidden_dim=100,
                 num_layers=1,
                 num_attention_heads=32,
                 dropout_prob = 0,
                 residual_dropout_prob = 0,
                 attention_dropout_prob= 0,
                 use_positional_encoding=True)

        max_conv_kernel_size = 5
        conv_output_size = 16
        self.convolutions = []

        self.convolutions.append(
                nn.Sequential(
                    #nn.ConstantPad2d((0,i - 1,0, i - 1), 0), # this outputs [batch,1,unified_query_length + i - 1 ,unified_document_length + i - 1]
                    nn.Conv2d(kernel_size=1, in_channels=2, out_channels=1), # this outputs [batch,32,unified_query_length,unified_document_length]
                    #nn.MaxPool3d(kernel_size=(conv_output_size,1,1)) # this outputs [batch,1,unified_query_length,unified_document_length]
            ))

        for i in range(2, max_conv_kernel_size + 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.ConstantPad2d((0,i - 1,0, i - 1), 0), # this outputs [batch,1,unified_query_length + i - 1 ,unified_document_length + i - 1]
                    nn.Conv2d(kernel_size=i, in_channels=2, out_channels=conv_output_size), # this outputs [batch,32,unified_query_length,unified_document_length]
                    nn.MaxPool3d(kernel_size=(conv_output_size,1,1)) # this outputs [batch,1,unified_query_length,unified_document_length]
            ))
        self.convolutions = nn.ModuleList(self.convolutions) # register conv as part of the model

        self.dense = nn.Linear(n_kernels * len(self.convolutions), 1, bias=True)
        self.dense_mean = nn.Linear(n_kernels * len(self.convolutions), 1, bias=True)
        self.dense_comb = nn.Linear(2, 1, bias=False)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor,
                query_idfs: torch.Tensor, document_idfs: torch.Tensor) -> torch.Tensor:

        #
        # contextualization
        #
        query_embeddings_context = self.contextualization(query_embeddings,query_pad_oov_mask) * query_pad_oov_mask.unsqueeze(-1)
        document_embeddings_context = self.contextualization(document_embeddings,document_pad_oov_mask) * document_pad_oov_mask.unsqueeze(-1)

        query_embeddings_concat = torch.cat([query_embeddings,query_embeddings_context],dim=2) 
        document_embeddings_concat = torch.cat([document_embeddings,document_embeddings_context],dim=2) 


        #
        # similarity matrix
        # -------------------------------------------------------

        # create sim matrix shape: (batch, 1, query_max, doc_max) for the input of conv_2d
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings).unsqueeze(1)
        cosine_matrix_context = self.dot_module.forward(query_embeddings_context, document_embeddings_context).unsqueeze(1)
        #cosine_matrix_concat = self.cosine_module.forward(query_embeddings_concat, document_embeddings_concat).unsqueeze(1)

        combined_cosine_matrix = torch.tanh(torch.cat([cosine_matrix,cosine_matrix_context],dim=1))


        #
        #  n-gram convolutions
        # ----------------------------------------------
        conv_results = []
        #conv_results.append(torch.topk(cosine_matrix.squeeze(),k=self.kmax_pooling_size,sorted=True)[0])

        for conv in self.convolutions:
            conv_results.append(conv(combined_cosine_matrix))
            #cr_kmax_result = torch.topk(cr.squeeze(),k=self.kmax_pooling_size,sorted=True)[0]
            #conv_results.append(cr_kmax_result)

        all_conv_results = torch.cat(conv_results,dim=1).unsqueeze(-1)

        #
        # flatten all paths together & weight by query idf
        # -------------------------------------------------------
        
        #per_query_results = torch.cat(conv_results,dim=-1)

       # weigthed_per_query = per_query_results * self.masked_softmax(query_idfs, query_pad_oov_mask.unsqueeze(-1))

        #all_flat = per_query_results.view(weigthed_per_query.shape[0],-1)

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1).unsqueeze(1)

        raw_kernel_results = torch.exp(- torch.pow(all_conv_results - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #
        # mean kernels
        #
        #kernel_results_masked2 = kernel_results_masked.clone()

        doc_lengths = torch.sum(document_pad_oov_mask, 1)
        #query_idf_softmax = self.masked_softmax(query_idfs, query_pad_oov_mask.unsqueeze(-1)).unsqueeze(1)
        #kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 3) #* query_idf_softmax
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10))
        #log_per_kernel_query_masked = log_per_kernel_query * query_idf_softmax
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1).unsqueeze(1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 2) 

        #per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)

        per_kernel_query_mean = per_kernel_query / doc_lengths.view(-1,1,1,1)

        log_per_kernel_query_mean = torch.log(torch.clamp(per_kernel_query_mean, min=1e-10))
        #log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_idf_softmax
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1).unsqueeze(1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 2) 


        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel.view(per_kernel.shape[0],-1))
        dense_mean_out = self.dense_mean(per_kernel_mean.view(per_kernel_mean.shape[0],-1))
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out],dim=1))
        score = torch.squeeze(dense_comb_out,1) #torch.tanh(dense_out), 1)
        return score

    
    def get_param_stats(self):
        return "MM_vnext_v1: dense w: "+str(self.dense.weight.data)+" b: "+str(self.dense.bias.data) +\
        "dense_mean weight: "+str(self.dense_mean.weight.data)+"b: "+str(self.dense_mean.bias.data) +\
        "dense_comb weight: "+str(self.dense_comb.weight.data) #+ "scaler: "+str(self.nn_scaler.data)

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


class Matchmaker_vNext_v2(nn.Module):
    '''
    
    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Matchmaker_vNext_v2(word_embeddings_out_dim)

    def __init__(self,
                 _embsize:int):
                 
        super(Matchmaker_vNext_v2,self).__init__()

        n_kernels = 11
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)

        self.cosine_module = CosineMatrixAttention()
        self.dot_module = DotProductMatrixAttention()
        self.masked_softmax = MaskedSoftmax()

        self.contextualization = StackedSelfAttentionEncoder(input_dim=_embsize,
                 hidden_dim=_embsize,
                 projection_dim=64,
                 feedforward_hidden_dim=100,
                 num_layers=1,
                 num_attention_heads=32,
                 dropout_prob = 0,
                 residual_dropout_prob = 0,
                 attention_dropout_prob= 0,
                 use_positional_encoding=True)

        max_conv_kernel_size = 5
        conv_output_size = 16
        self.convolutions = []

        self.convolutions.append(
                nn.Sequential(
                    #nn.ConstantPad2d((0,i - 1,0, i - 1), 0), # this outputs [batch,1,unified_query_length + i - 1 ,unified_document_length + i - 1]
                    nn.Conv2d(kernel_size=1, in_channels=2, out_channels=1), # this outputs [batch,32,unified_query_length,unified_document_length]
                    #nn.MaxPool3d(kernel_size=(conv_output_size,1,1)) # this outputs [batch,1,unified_query_length,unified_document_length]
            ))

        for i in range(2, max_conv_kernel_size + 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.ConstantPad2d((0,i - 1,0, i - 1), 0), # this outputs [batch,1,unified_query_length + i - 1 ,unified_document_length + i - 1]
                    nn.Conv2d(kernel_size=i, in_channels=2, out_channels=conv_output_size), # this outputs [batch,32,unified_query_length,unified_document_length]
                    nn.MaxPool3d(kernel_size=(conv_output_size,1,1)) # this outputs [batch,1,unified_query_length,unified_document_length]
            ))
        self.convolutions = nn.ModuleList(self.convolutions) # register conv as part of the model

        self.dense = nn.Linear(n_kernels * len(self.convolutions), 1, bias=True)
        self.dense_mean = nn.Linear(n_kernels * len(self.convolutions), 1, bias=True)
        self.dense_comb = nn.Linear(2, 1, bias=False)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor,
                query_idfs: torch.Tensor, document_idfs: torch.Tensor) -> torch.Tensor:

        #
        # contextualization
        #
        query_embeddings_context = self.contextualization(query_embeddings,query_pad_oov_mask) * query_pad_oov_mask.unsqueeze(-1)
        document_embeddings_context = self.contextualization(document_embeddings,document_pad_oov_mask) * document_pad_oov_mask.unsqueeze(-1)

        query_embeddings_concat = torch.cat([query_embeddings,query_embeddings_context],dim=2) 
        document_embeddings_concat = torch.cat([document_embeddings,document_embeddings_context],dim=2) 


        #
        # similarity matrix
        # -------------------------------------------------------

        # create sim matrix shape: (batch, 1, query_max, doc_max) for the input of conv_2d
        cosine_matrix = self.cosine_module.forward(query_embeddings, document_embeddings).unsqueeze(1)
        cosine_matrix_context = self.dot_module.forward(query_embeddings_context, document_embeddings_context).unsqueeze(1)
        #cosine_matrix_concat = self.cosine_module.forward(query_embeddings_concat, document_embeddings_concat).unsqueeze(1)

        combined_cosine_matrix = torch.tanh(torch.cat([cosine_matrix,cosine_matrix_context],dim=1))


        #
        #  n-gram convolutions
        # ----------------------------------------------
        conv_results = []
        #conv_results.append(torch.topk(cosine_matrix.squeeze(),k=self.kmax_pooling_size,sorted=True)[0])

        for conv in self.convolutions:
            conv_results.append(conv(combined_cosine_matrix))
            #cr_kmax_result = torch.topk(cr.squeeze(),k=self.kmax_pooling_size,sorted=True)[0]
            #conv_results.append(cr_kmax_result)

        all_conv_results = torch.cat(conv_results,dim=1).unsqueeze(-1)

        #
        # flatten all paths together & weight by query idf
        # -------------------------------------------------------
        
        #per_query_results = torch.cat(conv_results,dim=-1)

       # weigthed_per_query = per_query_results * self.masked_softmax(query_idfs, query_pad_oov_mask.unsqueeze(-1))

        #all_flat = per_query_results.view(weigthed_per_query.shape[0],-1)

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1).unsqueeze(1)

        raw_kernel_results = torch.exp(- torch.pow(all_conv_results - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        #
        # mean kernels
        #
        #kernel_results_masked2 = kernel_results_masked.clone()

        doc_lengths = torch.sum(document_pad_oov_mask, 1)
        #query_idf_softmax = self.masked_softmax(query_idfs, query_pad_oov_mask.unsqueeze(-1)).unsqueeze(1)
        #kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)

        per_kernel_query = torch.sum(kernel_results_masked, 3) #* query_idf_softmax
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10))
        #log_per_kernel_query_masked = log_per_kernel_query * query_idf_softmax
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1).unsqueeze(1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 2) 

        #per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)

        per_kernel_query_mean = per_kernel_query / doc_lengths.view(-1,1,1,1)

        log_per_kernel_query_mean = torch.log(torch.clamp(per_kernel_query_mean, min=1e-10))
        #log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_idf_softmax
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1).unsqueeze(1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 2) 


        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel.view(per_kernel.shape[0],-1))
        dense_mean_out = self.dense_mean(per_kernel_mean.view(per_kernel_mean.shape[0],-1))
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out],dim=1))
        score = torch.squeeze(dense_comb_out,1) #torch.tanh(dense_out), 1)
        return score

    
    def get_param_stats(self):
        return "MM_vnext_v1: dense w: "+str(self.dense.weight.data)+" b: "+str(self.dense.bias.data) +\
        "dense_mean weight: "+str(self.dense_mean.weight.data)+"b: "+str(self.dense_mean.bias.data) +\
        "dense_comb weight: "+str(self.dense_comb.weight.data) #+ "scaler: "+str(self.nn_scaler.data)

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

        

class Matchmaker_vNext_v3(nn.Module):
    '''
    Paper: ...

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return Matchmaker_vNext_v3(word_embeddings_out_dim, n_kernels = config["mm_light_kernels"])

    def __init__(self,
                 _embsize:int,
                 n_kernels: int):

        super(Matchmaker_vNext_v3, self).__init__()

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(self.kernel_mus(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(self.kernel_sigmas(n_kernels)), requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 10, dtype=torch.float32, requires_grad=True))
        self.nn_scaler2 = nn.Parameter(torch.full([1], 0.001, dtype=torch.float32, requires_grad=True))
        self.mixer = nn.Parameter(torch.full([1,1,1], 0.5, dtype=torch.float32, requires_grad=True))

        self.embedding_context = StackedSelfAttentionEncoder(input_dim=_embsize,
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

        self.interaction_squeezer = nn.Linear(_embsize, 124, bias=True)

        self.interaction_encoder = StackedSelfAttentionEncoder(input_dim=_embsize,
                 hidden_dim=_embsize,
                 projection_dim=32,
                 feedforward_hidden_dim=32,
                 num_layers=2,
                 num_attention_heads=16,
                 dropout_prob = 0,
                 residual_dropout_prob = 0,
                 attention_dropout_prob= 0,
                 #use_positional_encoding=False
                 )

        self.interaction_scorer = nn.Linear(self.interaction_encoder.get_output_dim(), 10, bias=True)
        self.interaction_scorer2 = nn.Linear(10, 1, bias=True)

        self.interaction_cls = nn.Parameter(torch.empty((1,1,self.interaction_encoder.get_input_dim())),requires_grad=True)
        torch.nn.init.uniform_(self.interaction_cls, -0.014, 0.014)

        self.interaction_mask_add = torch.full([1,1], 1, dtype=torch.float32, requires_grad=False).cuda()

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        #self.salc_W1 = nn.Linear(100, 1, bias=True)
        #self.salc_W2 = nn.Linear(_embsize, 100, bias=True)

        #self.masked_softmax = MaskedSoftmax()

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=True)
        self.dense_mean = nn.Linear(n_kernels, 1, bias=True)
        self.dense_comb = nn.Linear(3, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        torch.nn.init.uniform_(self.dense_mean.weight, -0.014, 0.014)  # inits taken from matchzoo

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor,
                query_idfs: torch.Tensor, document_idfs: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ

        query_embeddings = query_embeddings * query_pad_oov_mask.unsqueeze(-1) #* 10
        document_embeddings = document_embeddings * document_pad_oov_mask.unsqueeze(-1) #* 10

        query_embeddings_context = self.embedding_context(query_embeddings,query_pad_oov_mask)
        document_embeddings_context = self.embedding_context(document_embeddings,document_pad_oov_mask)

        query_embeddings_merged = (self.mixer * query_embeddings + (1 - self.mixer) * query_embeddings_context) * query_pad_oov_mask.unsqueeze(-1)
        document_embeddings_merged = (self.mixer * document_embeddings + (1 - self.mixer) * document_embeddings_context) * document_pad_oov_mask.unsqueeze(-1)

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(query_pad_oov_mask.unsqueeze(-1), document_pad_oov_mask.unsqueeze(-1).transpose(-1, -2))
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------


        # shape: (batch, query_max, doc_max)
        cosine_matrix = self.cosine_module.forward(query_embeddings_merged, document_embeddings_merged)
        cosine_matrix_masked = torch.tanh(cosine_matrix * query_by_doc_mask)
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        #
        # diff - pooling
        #

        max_per_doc = torch.max(cosine_matrix_masked, dim=1)
        max_per_doc_focused = torch.exp(- torch.pow(max_per_doc[0] - 1, 2) / (2 * (0.1 * 0.1)))

        #flat_qs = query_embeddings_merged.view(-1,query_embeddings_merged.shape[2])
        #index_offset= max_per_doc[1] + torch.arange(0,max_per_doc[1].shape[0]*query_embeddings.shape[1],query_embeddings.shape[1],device=max_per_doc[1].device).unsqueeze(-1)
        #max_queries = flat_qs.index_select(dim=0,index=index_offset.view(-1)).view(max_per_doc[1].shape[0],max_per_doc[1].shape[1],-1)

        merge_queries = torch.sum(query_embeddings_merged.unsqueeze(2).expand(-1,-1,cosine_matrix_masked.shape[2],-1) * masked_softmax(cosine_matrix_masked,query_by_doc_mask,dim=1).unsqueeze(-1),dim=1)

        diff_vectors =  torch.tanh(merge_queries - document_embeddings_merged) #document_embeddings_merged - merge_queries #

        diff_vectors_focused = diff_vectors * (torch.exp(max_per_doc_focused.unsqueeze(-1) * 2)-1) #self.interaction_squeezer(diff_vectors) #* self.nn_scaler  #* torch.softmax(max_per_doc[0],dim=1).unsqueeze(-1) #* max_per_doc_focused.unsqueeze(-1)

        diff_vectors_masked = diff_vectors_focused * document_pad_oov_mask.unsqueeze(-1)

        diff_vectors_masked_cls = torch.cat([self.interaction_cls.expand(diff_vectors_masked.shape[0],1,-1),diff_vectors_masked],dim=1)
        document_pad_oov_mask_cls = torch.cat([self.interaction_mask_add.expand(document_pad_oov_mask.shape[0],-1),document_pad_oov_mask],dim=1)

        encoded_diff = self.interaction_encoder(diff_vectors_masked_cls, document_pad_oov_mask_cls)
        encoded_pooled = encoded_diff[:,0] # like the bert cls token thing , add cls vector on first pos? +mask change?
        encoded_score2 = self.interaction_scorer(encoded_pooled)
        encoded_score = self.interaction_scorer2(encoded_score2) #/ torch.sum(document_pad_oov_mask, 1).unsqueeze(-1) #* self.nn_scaler2

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
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10))# * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        #per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)

        per_kernel_query_mean = per_kernel_query / doc_lengths.view(-1,1,1)

        log_per_kernel_query_mean = torch.log(torch.clamp(per_kernel_query_mean, min=1e-10))# * self.nn_scaler
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 



        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        dense_mean_out = self.dense_mean(per_kernel_mean)
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out,encoded_score],dim=1))
        score = torch.squeeze(encoded_score,1) #torch.tanh(dense_out), 1)
        return score

    def get_param_stats(self):
        return "MM_vnext_v3: dense w: "+str(self.dense.weight.data)+" b: "+str(self.dense.bias.data) +\
        "dense_mean weight: "+str(self.dense_mean.weight.data)+"b: "+str(self.dense_mean.bias.data) +\
        "dense_comb weight: "+str(self.dense_comb.weight.data) + "scaler: "+str(self.nn_scaler.data)+ "mixer: "+str(self.mixer.data)

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