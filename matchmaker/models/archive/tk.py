from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention                          
from allennlp.modules.matrix_attention.dot_product_matrix_attention import *                          
from allennlp.modules.seq2seq_encoders import PytorchTransformer
import math

class TK_v1(nn.Module):
    '''
    TK is a neural IR model - a fusion between transformer contextualization & kernel-based scoring

    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return TK_v1(word_embeddings_out_dim, 
                     kernels_mu =    config["tk_kernels_mu"],
                     kernels_sigma = config["tk_kernels_sigma"],
                     att_heads =     config["tk_att_heads"],
                     att_layer =     config["tk_att_layer"],
                     att_proj_dim =  config["tk_att_proj_dim"],
                     att_ff_dim =    config["tk_att_ff_dim"])

    def __init__(self,
                 _embsize:int,
                 kernels_mu: List[float],
                 kernels_sigma: List[float],
                 att_heads: int,
                 att_layer: int,
                 att_proj_dim: int,
                 att_ff_dim: int):

        super(TK_v1, self).__init__()

        n_kernels = len(kernels_mu)

        if len(kernels_mu) != len(kernels_sigma):
            raise Exception("len(kernels_mu) != len(kernels_sigma)")

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(kernels_mu), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(kernels_sigma), requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))
        self.mixer = nn.Parameter(torch.full([1,1,1], 0.5, dtype=torch.float32, requires_grad=True))

        self.stacked_att = PytorchTransformer(input_dim=_embsize,
                 hidden_dim=_embsize,
                 projection_dim=att_proj_dim,
                 feedforward_hidden_dim=att_ff_dim,
                 num_layers=att_layer,
                 num_attention_heads=att_heads,
                 dropout_prob = 0,
                 residual_dropout_prob = 0,
                 attention_dropout_prob = 0)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=False)
        self.dense_mean = nn.Linear(n_kernels, 1, bias=False)
        self.dense_comb = nn.Linear(2, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        torch.nn.init.uniform_(self.dense_mean.weight, -0.014, 0.014)  # inits taken from matchzoo

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor, 
                output_secondary_output: bool = False) -> torch.Tensor:
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
        log_per_kernel_query = torch.log2(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        #per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)

        per_kernel_query_mean = per_kernel_query / (doc_lengths.view(-1,1,1) + 1) # well, that +1 needs an explanation, sometimes training data is just broken ... (and nans all the things!)

        log_per_kernel_query_mean = per_kernel_query_mean * self.nn_scaler
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 


        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        dense_mean_out = self.dense_mean(per_kernel_mean)
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out],dim=1))
        score = torch.squeeze(dense_comb_out,1) #torch.tanh(dense_out), 1)

        if output_secondary_output:
            query_mean_vector = query_embeddings.sum(dim=1) / query_pad_oov_mask.sum(dim=1).unsqueeze(-1)
            return score, {"score":score,"dense_out":dense_out,"dense_mean_out":dense_mean_out,"per_kernel":per_kernel,
                           "per_kernel_mean":per_kernel_mean,"query_mean_vector":query_mean_vector,"cosine_matrix_masked":cosine_matrix_masked}
        else:
            return score

    def forward_representation(self, sequence_embeddings: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        seq_embeddings = sequence_embeddings * sequence_mask.unsqueeze(-1)
        seq_embeddings_context = self.stacked_att(sequence_embeddings, sequence_mask)
        seq_embeddings = (self.mixer * sequence_embeddings + (1 - self.mixer) * seq_embeddings_context) * sequence_mask.unsqueeze(-1)
        return seq_embeddings

    def get_param_stats(self): #" b: "+str(self.dense.bias.data) +\ "b: "+str(self.dense_mean.bias.data) +
        return "MM_light_v4b: dense w: "+str(self.dense.weight.data)+\
        "dense_mean weight: "+str(self.dense_mean.weight.data)+\
        "dense_comb weight: "+str(self.dense_comb.weight.data) + "scaler: "+str(self.nn_scaler.data) +"mixer: "+str(self.mixer.data)

    def get_param_secondary(self):
        return {"dense_weight":self.dense.weight,#"dense_bias":self.dense.bias,
                "dense_mean_weight":self.dense_mean.weight,#"dense_mean_bias":self.dense_mean.bias,
                "dense_comb_weight":self.dense_comb.weight, 
                "scaler":self.nn_scaler ,"mixer":self.mixer}


#    def kernel_mus(self, n_kernels: int):
#        """
#        get the mu for each guassian kernel. Mu is the middle of each bin
#        :param n_kernels: number of kernels (including exact match). first one is exact match
#        :return: l_mu, a list of mu.
#        """
#        l_mu = [1.0]
#        if n_kernels == 1:
#            return l_mu
#
#        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
#        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
#        for i in range(1, n_kernels - 1):
#            l_mu.append(l_mu[i] - bin_size)
#        return l_mu
#
#    def kernel_sigmas(self, n_kernels: int):
#        """
#        get sigmas for each guassian kernel.
#        :param n_kernels: number of kernels (including exactmath.)
#        :param lamb:
#        :param use_exact:
#        :return: l_sigma, a list of simga
#        """
#        bin_size = 2.0 / (n_kernels - 1)
#        #l_sigma = [0.0001]  # for exact match. small variance -> exact match
#        #if n_kernels == 1:
#        #    return l_sigma
#
#        l_sigma = [0.5 * bin_size] * (n_kernels)
#        return l_sigma


class TK_v2(nn.Module):
    '''
    TK is a neural IR model - a fusion between transformer contextualization & kernel-based scoring

    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):

        ws = [20,30,50,80,100,120,150]
        max_windows = [math.ceil(config["max_doc_length"] / float(w)) for w in ws]

        return TK_v2(word_embeddings_out_dim, 
                     kernels_mu =    config["tk_kernels_mu"],
                     kernels_sigma = config["tk_kernels_sigma"],
                     att_heads =    config["tk_att_heads"],
                     att_layer =    config["tk_att_layer"],
                     att_proj_dim = config["tk_att_proj_dim"],
                     att_ff_dim =   config["tk_att_ff_dim"],
                     win_size = ws,
                     max_windows = max_windows)

    def __init__(self,
                 _embsize:int,
                 kernels_mu: List[float],
                 kernels_sigma: List[float],
                 att_heads: int,
                 att_layer: int,
                 att_proj_dim: int,
                 att_ff_dim: int,
                 win_size:int,
                 max_windows:int):

        super(TK_v2, self).__init__()

        n_kernels = len(kernels_mu)

        if len(kernels_mu) != len(kernels_sigma):
            raise Exception("len(kernels_mu) != len(kernels_sigma)")


        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(kernels_mu), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(kernels_sigma), requires_grad=False).view(1, 1, 1, n_kernels)
        self.mixer = nn.Parameter(torch.full([1,1,1], 0.5, dtype=torch.float32, requires_grad=True))

        self.stacked_att = PytorchTransformer(input_dim=_embsize,
                 hidden_dim=_embsize,
                 projection_dim=att_proj_dim,
                 feedforward_hidden_dim=att_ff_dim,
                 num_layers=att_layer,
                 num_attention_heads=att_heads,
                 dropout_prob = 0,
                 residual_dropout_prob = 0,
                 attention_dropout_prob = 0)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        # bias is set to True in original code (we found it to not help, how could it?)
        #self.dense_mean = nn.Linear(n_kernels, 1, bias=True)

        self.nn_scaler = nn.ParameterList([nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True)) for w in win_size])

        self.kernel_weights = nn.ModuleList([nn.Linear(n_kernels, 1, bias=False) for w in win_size])
        
        self.window_size = win_size
        self.window_scorer = []
        for w in max_windows:
            l =  nn.Linear(w, 1, bias=False)
            torch.nn.init.constant_(l.weight, 1/w)
            self.window_scorer.append(l)

        self.window_scorer = nn.ModuleList(self.window_scorer)

        self.window_merger = nn.Linear(len(self.window_size), 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        #torch.nn.init.uniform_(self.dense.weight, 0.001, 0.034)  # inits taken from matchzoo
        #torch.nn.init.uniform_(self.dense_mean.weight, -0.014, 0.014)  # inits taken from matchzoo

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        #torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor, 
                output_secondary_output: bool = False) -> torch.Tensor:
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


        #"dense_mean weight: "+str(self.dense_mean.weight.data)+"b: "+str(self.dense_mean.bias.data) + "dense_comb weight: "+str(self.dense_comb.weight.data) + +" b: "+str(self.dense.bias.data) +\
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


#    def kernel_mus(self, n_kernels: int):
#        """
#        get the mu for each guassian kernel. Mu is the middle of each bin
#        :param n_kernels: number of kernels (including exact match). first one is exact match
#        :return: l_mu, a list of mu.
#        """
#        l_mu = [1.0]
#        if n_kernels == 1:
#            return l_mu
#
#        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
#        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
#        for i in range(1, n_kernels - 1):
#            l_mu.append(l_mu[i] - bin_size)
#        return l_mu
#
#    def kernel_sigmas(self, n_kernels: int):
#        """
#        get sigmas for each guassian kernel.
#        :param n_kernels: number of kernels (including exactmath.)
#        :param lamb:
#        :param use_exact:
#        :return: l_sigma, a list of simga
#        """
#        bin_size = 2.0 / (n_kernels - 1)
#        #l_sigma = [0.0001]  # for exact match. small variance -> exact match
#        #if n_kernels == 1:
#        #    return l_sigma
##
#        l_sigma = [0.5 * bin_size] * (n_kernels)
#        return l_sigma


class TK_v3(nn.Module):
    '''
    TK is a neural IR model - a fusion between transformer contextualization & kernel-based scoring

    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return TK_v3(word_embeddings_out_dim, 
                     kernels_mu =    config["tk_kernels_mu"],
                     kernels_sigma = config["tk_kernels_sigma"],
                     att_heads =     config["tk_att_heads"],
                     att_layer =     config["tk_att_layer"],
                     att_proj_dim =  config["tk_att_proj_dim"],
                     att_ff_dim =    config["tk_att_ff_dim"])

    def __init__(self,
                 _embsize:int,
                 kernels_mu: List[float],
                 kernels_sigma: List[float],
                 att_heads: int,
                 att_layer: int,
                 att_proj_dim: int,
                 att_ff_dim: int):

        super(TK_v3, self).__init__()

        n_kernels = len(kernels_mu)

        if len(kernels_mu) != len(kernels_sigma):
            raise Exception("len(kernels_mu) != len(kernels_sigma)")

        # static - kernel size & magnitude variables
        #self.mu = Variable(torch.cuda.FloatTensor(kernels_mu), requires_grad=False).view(1, 1, 1, n_kernels)
        #self.sigma = Variable(torch.cuda.FloatTensor(kernels_sigma), requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))
        self.mixer = nn.Parameter(torch.full([1,1,1], 0.5, dtype=torch.float32, requires_grad=True))

        self.stacked_att = PytorchTransformer(input_dim=_embsize,
                 hidden_dim=_embsize,
                 projection_dim=att_proj_dim,
                 feedforward_hidden_dim=att_ff_dim,
                 num_layers=att_layer,
                 num_attention_heads=att_heads,
                 dropout_prob = 0,
                 residual_dropout_prob = 0,
                 attention_dropout_prob = 0)

        self.stacked_att_scoring = PytorchTransformer(input_dim=1,
                 hidden_dim=32,
                 projection_dim=32,
                 feedforward_hidden_dim=32,
                 num_layers=3,
                 num_attention_heads=8,
                 dropout_prob = 0,
                 residual_dropout_prob = 0,
                 attention_dropout_prob = 0,
                 use_positional_encoding=True)

        self.stacked_att_scoring_cls_token = nn.Parameter(torch.full([1,1], 0.5, dtype=torch.float32, requires_grad=True))
        self.stacked_att_scoring_cls_mask = torch.full([1,1], 1, dtype=torch.float32, requires_grad=False).cuda()

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        # bias is set to True in original code (we found it to not help, how could it?)
        self.stacked_att_scoring_funnel = nn.Linear(self.stacked_att_scoring.get_output_dim(), 1, bias=False)
        #self.dense_mean = nn.Linear(n_kernels, 1, bias=False)
        #self.dense_comb = nn.Linear(2, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        #torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #torch.nn.init.uniform_(self.dense_mean.weight, -0.014, 0.014)  # inits taken from matchzoo

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        #torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor, 
                output_secondary_output: bool = False) -> torch.Tensor:
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
        cosine_matrix_masked = cosine_matrix * query_by_doc_mask

        #cls_cosine = torch.cat([self.stacked_att_scoring_cls_token.expand(cosine_matrix_masked.shape[0],1,1),cosine_matrix_masked.view(cosine_matrix_masked.shape[0],-1,1)],dim=1)
        #cls_cosine_mask = torch.cat([self.stacked_att_scoring_cls_mask.expand(cosine_matrix_masked.shape[0],1,1),query_by_doc_mask.view(query_by_doc_mask.shape[0],-1,1)],dim=1)

        cls_cosine = torch.cat([self.stacked_att_scoring_cls_token.expand(cosine_matrix_masked.shape[0],cosine_matrix_masked.shape[1],1),cosine_matrix_masked],dim=2)
        cls_cosine_mask = torch.cat([self.stacked_att_scoring_cls_mask.expand(cosine_matrix_masked.shape[0],cosine_matrix_masked.shape[1],1),query_by_doc_mask],dim=2)

        scoring_result = []

        for i in range(cls_cosine.shape[1]):
            scoring_result.append(self.stacked_att_scoring(cls_cosine[:,i,:].unsqueeze(-1),cls_cosine_mask[:,i,:])[:,0].unsqueeze(1))
        #score = scoring_result[:,0]

        
        score = self.stacked_att_scoring_funnel(torch.cat(scoring_result,dim=1)).squeeze(-1)
        #score = torch.mean(score,dim=-1)
        score = torch.sum(score * query_pad_oov_mask,dim=-1) / torch.sum(query_pad_oov_mask,dim=-1)
        #cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)
#
#
        #self.stacked_att_scoring
        ##
        ## gaussian kernels & soft-TF
        ##
        ## first run through kernel, then sum on doc dim then sum on query dim
        ## -------------------------------------------------------
        #
        #raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        #kernel_results_masked = raw_kernel_results * query_by_doc_mask_view
#
        ##
        ## mean kernels
        ##
        ##kernel_results_masked2 = kernel_results_masked.clone()
#
        #doc_lengths = torch.sum(document_pad_oov_mask, 1)
#
        ##kernel_results_masked2_mean = kernel_results_masked / doc_lengths.unsqueeze(-1)
#
        #per_kernel_query = torch.sum(kernel_results_masked, 2)
        #log_per_kernel_query = torch.log2(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        #log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        #per_kernel = torch.sum(log_per_kernel_query_masked, 1) 
#
        ##per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)
#
        #per_kernel_query_mean = per_kernel_query / (doc_lengths.view(-1,1,1) + 1) # well, that +1 needs an explanation, sometimes training data is just broken ... (and nans all the things!)
#
        #log_per_kernel_query_mean = per_kernel_query_mean * self.nn_scaler
        #log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        #per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 
#
#
        ###
        ### "Learning to rank" layer - connects kernels with learned weights
        ### -------------------------------------------------------
#
        #dense_out = self.dense(per_kernel)
        #dense_mean_out = self.dense_mean(per_kernel_mean)
        #dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out],dim=1))
        #score = torch.squeeze(dense_comb_out,1) #torch.tanh(dense_out), 1)

        if output_secondary_output:
            query_mean_vector = query_embeddings.sum(dim=1) / query_pad_oov_mask.sum(dim=1).unsqueeze(-1)
            return score, {"score":score,"query_mean_vector":query_mean_vector,"cosine_matrix_masked":cosine_matrix_masked}
        else:
            return score

    def get_param_stats(self): #" b: "+str(self.dense.bias.data) +\ "b: "+str(self.dense_mean.bias.data) +
        return "MM_light_v4b: scaler: "+str(self.nn_scaler.data) +"mixer: "+str(self.mixer.data)

    def get_param_secondary(self):
        return {#"dense_weight":self.dense.weight,#"dense_bias":self.dense.bias,
                #"dense_mean_weight":self.dense_mean.weight,#"dense_mean_bias":self.dense_mean.bias,
                #"dense_comb_weight":self.dense_comb.weight, 
                "scaler":self.nn_scaler ,"mixer":self.mixer}

class TK_v4_pos(nn.Module):
    '''
    TK is a neural IR model - a fusion between transformer contextualization & kernel-based scoring

    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return TK_v4_pos(word_embeddings_out_dim, 
                     kernels_mu =    config["tk_kernels_mu"],
                     kernels_sigma = config["tk_kernels_sigma"],
                     att_heads =     config["tk_att_heads"],
                     att_layer =     config["tk_att_layer"],
                     att_proj_dim =  config["tk_att_proj_dim"],
                     att_ff_dim =    config["tk_att_ff_dim"],
                     max_length = config["max_doc_length"])

    def __init__(self,
                 _embsize:int,
                 kernels_mu: List[float],
                 kernels_sigma: List[float],
                 att_heads: int,
                 att_layer: int,
                 att_proj_dim: int,
                 att_ff_dim: int,
                 max_length):

        super(TK_v4_pos, self).__init__()

        n_kernels = len(kernels_mu)

        if len(kernels_mu) != len(kernels_sigma):
            raise Exception("len(kernels_mu) != len(kernels_sigma)")

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(kernels_mu), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(kernels_sigma), requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))
        self.pos_neg_scaler = nn.Parameter(torch.full([1], 1, dtype=torch.float32, requires_grad=True))
        #self.pos_neg_scaler_q = nn.Parameter(torch.full([1], 0.8, dtype=torch.float32, requires_grad=True))
        self.mixer = nn.Parameter(torch.full([1,1,1], 0.5, dtype=torch.float32, requires_grad=True))

        pos_f = self.get_positional_features(_embsize,max_length)
        pos_f.requires_grad = False
        self.positional_features = pos_f #nn.Parameter(pos_f)
        self.positional_features.requires_grad = False

        #self.positional_features_q = self.get_positional_features(_embsize,max_length)
        #self.positional_features_q.requires_grad = True


        self.stacked_att = PytorchTransformer(input_dim=_embsize,
                 hidden_dim=_embsize,
                 projection_dim=att_proj_dim,
                 feedforward_hidden_dim=att_ff_dim,
                 num_layers=att_layer,
                 num_attention_heads=att_heads,
                 dropout_prob = 0,
                 residual_dropout_prob = 0,
                 attention_dropout_prob = 0,
                 use_positional_encoding=False)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=False)
        self.dense_mean = nn.Linear(n_kernels, 1, bias=False)
        self.dense_comb = nn.Linear(2, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        torch.nn.init.uniform_(self.dense_mean.weight, -0.014, 0.014)  # inits taken from matchzoo

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor, 
                output_secondary_output: bool = False) -> torch.Tensor:
        # pylint: disable=arguments-differ

        query_embeddings = query_embeddings * query_pad_oov_mask.unsqueeze(-1)
        document_embeddings = document_embeddings * document_pad_oov_mask.unsqueeze(-1)

        query_embeddings_context = self.stacked_att(query_embeddings + self.positional_features[:,:query_embeddings.shape[1],:],query_pad_oov_mask) - \
                                    (self.stacked_att(self.positional_features[:,:query_embeddings.shape[1],:].expand(query_embeddings.shape[0],-1,-1),query_pad_oov_mask) ) #* self.pos_neg_scaler
                                    
        document_embeddings_context = self.stacked_att(document_embeddings + self.positional_features[:,:document_embeddings.shape[1],:],document_pad_oov_mask) - \
                                    (self.stacked_att(self.positional_features[:,:document_embeddings.shape[1],:].expand(document_embeddings.shape[0],-1,-1),document_pad_oov_mask) ) #* self.pos_neg_scaler

        #query_embeddings = torch.cat([query_embeddings,query_embeddings_context],dim=2) * query_pad_oov_mask.unsqueeze(-1)
        #document_embeddings = torch.cat([document_embeddings,document_embeddings_context],dim=2) * document_pad_oov_mask.unsqueeze(-1)
        #query_embeddings = (self.mixer * query_embeddings + (1 - self.mixer) * query_embeddings_context) * query_pad_oov_mask.unsqueeze(-1)
       # document_embeddings = (self.mixer * document_embeddings + (1 - self.mixer) * document_embeddings_context) * document_pad_oov_mask.unsqueeze(-1)

        query_embeddings =  query_embeddings_context#) * query_pad_oov_mask.unsqueeze(-1)
        document_embeddings =  document_embeddings_context#) * document_pad_oov_mask.unsqueeze(-1)



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
        log_per_kernel_query = torch.log2(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        #per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)

        per_kernel_query_mean = per_kernel_query / (doc_lengths.view(-1,1,1) + 1) # well, that +1 needs an explanation, sometimes training data is just broken ... (and nans all the things!)

        log_per_kernel_query_mean = per_kernel_query_mean * self.nn_scaler
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 


        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        dense_mean_out = self.dense_mean(per_kernel_mean)
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out],dim=1))
        #score = torch.squeeze(dense_comb_out,1) #torch.tanh(dense_out), 1)
        score = torch.squeeze(dense_out,1) #torch.tanh(dense_out), 1)

        if output_secondary_output:
            query_mean_vector = query_embeddings.sum(dim=1) / query_pad_oov_mask.sum(dim=1).unsqueeze(-1)
            return score, {"score":score,"dense_out":dense_out,"dense_mean_out":dense_mean_out,"per_kernel":per_kernel,
                           "per_kernel_mean":per_kernel_mean,"query_mean_vector":query_mean_vector,"cosine_matrix_masked":cosine_matrix_masked}
        else:
            return score

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

    def forward_representation(self, sequence_embeddings: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        seq_embeddings = sequence_embeddings * sequence_mask.unsqueeze(-1)
        seq_embeddings_context = self.stacked_att(sequence_embeddings + self.positional_features[:,:sequence_embeddings.shape[1],:],sequence_mask) - \
                                    (self.stacked_att(self.positional_features[:,:sequence_embeddings.shape[1],:].expand(sequence_embeddings.shape[0],-1,-1),sequence_mask) * self.pos_neg_scaler )
        seq_embeddings = (self.mixer * sequence_embeddings + (1 - self.mixer) * seq_embeddings_context) * sequence_mask.unsqueeze(-1)
        return seq_embeddings

    def get_param_stats(self): #" b: "+str(self.dense.bias.data) +\ "b: "+str(self.dense_mean.bias.data) +
        return "MM_light_v4b: dense w: "+str(self.dense.weight.data)+\
        "dense_mean weight: "+str(self.dense_mean.weight.data)+\
        "dense_comb weight: "+str(self.dense_comb.weight.data) + "scaler: "+str(self.nn_scaler.data) + "pos_neg_scaler: "+str(self.pos_neg_scaler.data) +"mixer: "+str(self.mixer.data)

    def get_param_secondary(self):
        return {"dense_weight":self.dense.weight,#"dense_bias":self.dense.bias,
                "dense_mean_weight":self.dense_mean.weight,#"dense_mean_bias":self.dense_mean.bias,
                "dense_comb_weight":self.dense_comb.weight, 
                "scaler":self.nn_scaler ,"mixer":self.mixer}

from allennlp.modules.feedforward import FeedForward

from allennlp.nn.activations import Activation


class TK_v5(nn.Module):
    '''
    TK is a neural IR model - a fusion between transformer contextualization & kernel-based scoring

    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return TK_v5(word_embeddings_out_dim, 
                     kernels_mu =    config["tk_kernels_mu"],
                     kernels_sigma = config["tk_kernels_sigma"],
                     att_heads =     config["tk_att_heads"],
                     att_layer =     config["tk_att_layer"],
                     att_proj_dim =  config["tk_att_proj_dim"],
                     att_ff_dim =    config["tk_att_ff_dim"],
                     max_length = config["max_doc_length"])

    def __init__(self,
                 _embsize:int,
                 kernels_mu: List[float],
                 kernels_sigma: List[float],
                 att_heads: int,
                 att_layer: int,
                 att_proj_dim: int,
                 att_ff_dim: int,
                 max_length):

        super(TK_v5, self).__init__()

        n_kernels = len(kernels_mu)

        if len(kernels_mu) != len(kernels_sigma):
            raise Exception("len(kernels_mu) != len(kernels_sigma)")

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(kernels_mu), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(kernels_sigma), requires_grad=False).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))
        self.pos_neg_scaler = nn.Parameter(torch.full([1], 1, dtype=torch.float32, requires_grad=True))
        #self.pos_neg_scaler_q = nn.Parameter(torch.full([1], 0.8, dtype=torch.float32, requires_grad=True))

        pos_f = self.get_positional_features(_embsize,max_length,max_timescale=100)
        pos_f.requires_grad = False
        self.positional_features = pos_f #nn.Parameter(pos_f)
        self.positional_features.requires_grad = False

        #self.positional_features_q = self.get_positional_features(_embsize,max_length)
        #self.positional_features_q.requires_grad = True

        self.attention_layers = att_layer
        self.attention = []
        #self.attention_ff = []
        for i in range(self.attention_layers):
            self.attention.append(torch.nn.MultiheadAttention(_embsize, att_heads,bias=False))
            #self.attention_ff.append(FeedForward(_embsize,
            #                         activations=[Activation.by_name('relu')(),
            #                                      Activation.by_name('linear')()],
            #                         hidden_dims=[att_ff_dim, _embsize],
            #                         num_layers=2,
            #                         dropout=0))

        self.attention = nn.ModuleList(self.attention)

        self.mixer = nn.Parameter(torch.full([self.attention_layers + 1,1,1,1], 0.5, dtype=torch.float32, requires_grad=True))
        self.position_mixer = nn.Parameter(torch.full([1,1,1], 0.1, dtype=torch.float32, requires_grad=True))


        #self.attention_ff = nn.ModuleList(self.attention_ff)

        #self.stacked_att = PytorchTransformer(input_dim=_embsize,
        #         hidden_dim=_embsize,
        #         projection_dim=att_proj_dim,
        #         feedforward_hidden_dim=att_ff_dim,
        #         num_layers=att_layer,
        #         num_attention_heads=att_heads,
        #         dropout_prob = 0,
        #         residual_dropout_prob = 0,
        #         attention_dropout_prob = 0,
        #         use_positional_encoding=False)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=False)
        self.dense_mean = nn.Linear(n_kernels, 1, bias=False)
        self.dense_comb = nn.Linear(2, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        torch.nn.init.uniform_(self.dense_mean.weight, -0.014, 0.014)  # inits taken from matchzoo

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #self.dense.bias.data.fill_(0.0)

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor, 
                output_secondary_output: bool = False) -> torch.Tensor:
        # pylint: disable=arguments-differ

        query_embeddings = self.forward_representation(query_embeddings, query_pad_oov_mask)
        document_embeddings = self.forward_representation(document_embeddings, document_pad_oov_mask)

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
        log_per_kernel_query = torch.log2(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

        #per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)

        per_kernel_query_mean = per_kernel_query / (doc_lengths.view(-1,1,1) + 1) # well, that +1 needs an explanation, sometimes training data is just broken ... (and nans all the things!)

        log_per_kernel_query_mean = per_kernel_query_mean * self.nn_scaler
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1) 


        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        dense_mean_out = self.dense_mean(per_kernel_mean)
        dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out],dim=1))
        #score = torch.squeeze(dense_comb_out,1) #torch.tanh(dense_out), 1)
        score = torch.squeeze(dense_out,1) #torch.tanh(dense_out), 1)

        if output_secondary_output:
            query_mean_vector = query_embeddings.sum(dim=1) / query_pad_oov_mask.sum(dim=1).unsqueeze(-1)
            return score, {"score":score,"dense_out":dense_out,"dense_mean_out":dense_mean_out,"per_kernel":per_kernel,
                           "per_kernel_mean":per_kernel_mean,"query_mean_vector":query_mean_vector,"cosine_matrix_masked":cosine_matrix_masked}
        else:
            return score

    def forward_representation(self, sequence_embeddings: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:

        for i,attention in enumerate(self.attention):

            positional_sequence = (sequence_embeddings + self.positional_features[:,:sequence_embeddings.shape[1],:]).transpose(1,0)
            att_output, weights = attention.forward(query=positional_sequence,key=positional_sequence,value=sequence_embeddings.transpose(1,0),key_padding_mask=~sequence_mask.bool())

            sequence_embeddings = (self.mixer[i] * sequence_embeddings + (1 - self.mixer[i]) * att_output.transpose(1,0)) * sequence_mask.unsqueeze(-1)

        return sequence_embeddings

        #orig_seq = sequence_embeddings
#
        #for i,attention in enumerate(self.attention):
#
        #    positional_sequence = (sequence_embeddings + self.positional_features[:,:sequence_embeddings.shape[1],:]).transpose(1,0)
        #    att_output, weights = attention.forward(query=positional_sequence,key=positional_sequence,value=sequence_embeddings.transpose(1,0),key_padding_mask=~sequence_mask.bool())
#
        #    sequence_embeddings = (self.mixer[i] * sequence_embeddings + (1 - self.mixer[i]) * att_output.transpose(1,0)) * sequence_mask.unsqueeze(-1)
        #
        #sequence_embeddings = (self.mixer[-1] * orig_seq + (1 - self.mixer[-1]) * sequence_embeddings) * sequence_mask.unsqueeze(-1)
#
        #sequence_embeddings = self.position_mixer * self.positional_features[:,:sequence_embeddings.shape[1],:] + (1 - self.position_mixer) * sequence_embeddings
#
        #return sequence_embeddings


        #orig_seq = sequence_embeddings
#
        #positional_sequence = (sequence_embeddings + self.positional_features[:,:sequence_embeddings.shape[1],:]).transpose(1,0)
#
        #for i,attention in enumerate(self.attention):
#
        #    positional_sequence = self.attention_ff[i](positional_sequence)
        #    sequence_embeddings = self.attention_ff[i](sequence_embeddings)
#
        #    att_output_pos, _ = attention.forward(query=positional_sequence,key=positional_sequence,value=positional_sequence,key_padding_mask=~sequence_mask.bool())
        #    att_output_agnostic, _ = attention.forward(query=positional_sequence,key=positional_sequence,value=sequence_embeddings.transpose(1,0),key_padding_mask=~sequence_mask.bool())
#
        #    sequence_embeddings = att_output_agnostic.transpose(1,0) # (self.mixer * sequence_embeddings + (1 - self.mixer) * ) * sequence_mask.unsqueeze(-1)
        #    positional_sequence = att_output_pos#.transpose(1,0) # (self.mixer * sequence_embeddings + (1 - self.mixer) * ) * sequence_mask.unsqueeze(-1)
        #
        #sequence_embeddings = (self.mixer * orig_seq + (1 - self.mixer) * sequence_embeddings) * sequence_mask.unsqueeze(-1)
#
        #return sequence_embeddings

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

    

    def get_param_stats(self): #" b: "+str(self.dense.bias.data) +\ "b: "+str(self.dense_mean.bias.data) +
        return "MM_light_v4b: dense w: "+str(self.dense.weight.data)+\
        "dense_mean weight: "+str(self.dense_mean.weight.data)+\
        "dense_comb weight: "+str(self.dense_comb.weight.data) + "scaler: "+str(self.nn_scaler.data) + "position_mixer: "+str(self.position_mixer.data) +"mixer: "+str(self.mixer.data)

    def get_param_secondary(self):
        return {"dense_weight":self.dense.weight,#"dense_bias":self.dense.bias,
                "dense_mean_weight":self.dense_mean.weight,#"dense_mean_bias":self.dense_mean.bias,
                "dense_comb_weight":self.dense_comb.weight, 
                "scaler":self.nn_scaler ,"mixer":self.mixer}

from matchmaker.modules.pos_agnostic_transformer import TransformerEncoderLayerPosAgnostic, TransformerEncoderPosAgnostic
import random

class TK_v6(nn.Module):
    '''
    TK is a neural IR model - a fusion between transformer contextualization & kernel-based scoring

    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return TK_v6(word_embeddings_out_dim, 
                     kernels_mu =    config["tk_kernels_mu"],
                     kernels_sigma = config["tk_kernels_sigma"],
                     att_heads =     config["tk_att_heads"],
                     att_layer =     config["tk_att_layer"],
                     att_proj_dim =  config["tk_att_proj_dim"],
                     att_ff_dim =    config["tk_att_ff_dim"],
                     max_length =    config["max_doc_length"],
                     use_pos_agnostic     = config["tk_use_pos_agnostic"],
                     use_position_bias    = config["tk_use_position_bias"],  
                     use_diff_posencoding = config["tk_use_diff_posencoding"],
                     position_bias_bin_percent   = config["tk_position_bias_bin_percent"],
                     position_bias_absolute_steps= config["tk_position_bias_absolute_steps"] )

    def __init__(self,
                 _embsize:int,
                 kernels_mu: List[float],
                 kernels_sigma: List[float],
                 att_heads: int,
                 att_layer: int,
                 att_proj_dim: int,
                 att_ff_dim: int,
                 max_length,
                 use_pos_agnostic,  
                 use_position_bias,
                 use_diff_posencoding,
                 position_bias_bin_percent,
                 position_bias_absolute_steps):

        super(TK_v6, self).__init__()

        n_kernels = len(kernels_mu)
        self.use_pos_agnostic     = use_pos_agnostic    
        self.use_position_bias    = use_position_bias   
        self.use_diff_posencoding = use_diff_posencoding
        self.position_bias_bin_percent = position_bias_bin_percent
        self.position_bias_absolute_steps = position_bias_absolute_steps

        if len(kernels_mu) != len(kernels_sigma):
            raise Exception("len(kernels_mu) != len(kernels_sigma)")

        # static - kernel size & magnitude variables
        self.register_buffer("mu",nn.Parameter(torch.tensor(kernels_mu), requires_grad=False).view(1, 1, 1, n_kernels))
        self.register_buffer("sigma", nn.Parameter(torch.tensor(kernels_sigma), requires_grad=False).view(1, 1, 1, n_kernels))
        #self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))

        self.layer_norm = nn.LayerNorm(_embsize)

        pos_f = self.get_positional_features(_embsize,max_length) #max_timescale=100000
        pos_f.requires_grad = True
        self.positional_features_q = nn.Parameter(pos_f)
        self.positional_features_q.requires_grad = True

        if self.use_diff_posencoding == True:
            pos_f = self.get_positional_features(_embsize,max_length+500) #max_timescale=100000
            pos_f.requires_grad = True
            self.positional_features_d = nn.Parameter(pos_f[:,500:,:]) #nn.Parameter(pos_f)
            self.positional_features_d.requires_grad = True
        else:
            self.positional_features_d = self.positional_features_q


        self.mixer = nn.Parameter(torch.full([1], 0.5, dtype=torch.float32, requires_grad=True))


        self.position_bias = nn.Parameter(torch.full([int(1/self.position_bias_bin_percent) + 1], 1, dtype=torch.float32, requires_grad=True))
        self.position_bias.data[0] = 0 # padding
        #self.position_bias.data[1,:] = 1.3
        #self.position_bias.data[2,:] = 1.2
        #self.position_bias.data[3,:] = 1.1

        self.position_bias_absolute_factors = math.ceil(max_length/self.position_bias_absolute_steps)

        self.position_bias_index_selects = torch.tensor([i + 1 for i in range(self.position_bias_absolute_factors) for _ in range(self.position_bias_absolute_steps)]).float().cuda()
        self.position_bias_absolute = nn.Parameter(torch.full([self.position_bias_absolute_factors + 1], 1, dtype=torch.float32, requires_grad=True))
        self.position_bias_absolute.data[0] = 0 # padding
        #self.position_bias_absolute.data[1:4,:] = 1.3
        #self.position_bias_absolute.data[4:7,:] = 1.2
        #self.position_bias_absolute.data[7:11,:]= 1.1


        if self.use_pos_agnostic == True:
            encoder_layer = TransformerEncoderLayerPosAgnostic(_embsize, att_heads, dim_feedforward=att_ff_dim, dropout=0)
            self.contextualizer = TransformerEncoderPosAgnostic(encoder_layer, att_layer, norm=None)
        else:
            encoder_layer = nn.TransformerEncoderLayer(_embsize, att_heads, dim_feedforward=att_ff_dim, dropout=0) #,activation="gelu")
            self.contextualizer = nn.TransformerEncoder(encoder_layer, att_layer, norm=None) # nn.LayerNorm(_embsize)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        self.dense = nn.Linear(n_kernels, 1, bias=False)
        #torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014) # inits taken from matchzoo

    def clear_explicit_bias(self):
        self.position_bias.data[1:] = 1
        self.position_bias_absolute.data[1:] = 1

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor, 
                output_secondary_output: bool = False) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # contextualization
        # -------------------------------------------------------

        query_embeddings = self.forward_representation(query_embeddings, query_pad_oov_mask,"query")
        document_embeddings = self.forward_representation(document_embeddings, document_pad_oov_mask,"doc_model")

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
        # explicit position bias
        # -------------------------------------------------------

        original_kernel_results_masked = kernel_results_masked

        if self.use_position_bias == True:
    
            mask = torch.zeros_like(document_pad_oov_mask)
            pos_range = (torch.round(doc_lengths*self.position_bias_bin_percent).unsqueeze(-1) * torch.arange(0,self.position_bias.shape[0] - 1,device="cuda").float()).long()
            for b in range(mask.shape[0]):
                mask[b,pos_range[b]] = 1
            cum_sum = torch.cumsum(mask,dim = 1)
    
            pos_indices = (cum_sum * document_pad_oov_mask).long().view(-1)
            position_bias_map = torch.index_select(self.position_bias,0,pos_indices).view(mask.shape[0],1,-1,1)
    
            abs_indices = (self.position_bias_index_selects[:document_embeddings.shape[1]] * document_pad_oov_mask).long().view(-1)
            absolute_bias = torch.index_select(self.position_bias_absolute,0,abs_indices).view(mask.shape[0],1,-1,1)

            kernel_results_masked = kernel_results_masked * position_bias_map * absolute_bias 

        #
        # kernel-pooling
        # -------------------------------------------------------

        per_kernel_query = torch.sum(kernel_results_masked, 2) # k*
#        k = torch.sum(original_kernel_results_masked, 2) # k


        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) #* self.nn_scaler
#        k_log = torch.log(torch.clamp(k, min=1e-10)) #* self.nn_scaler



        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

#        k_log = k_log * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
#        per_kernel_k = torch.sum(k_log, 1) 
#
#        k_star_sum = torch.mean(per_kernel,1)
#        k_sum = torch.mean(per_kernel_k,1)
#
#        alpha = torch.exp((k_star_sum - k_sum) / torch.sum(query_pad_oov_mask, 1))
#
#        #print(alpha)
#        original_kernel_results_masked = original_kernel_results_masked * alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#        per_kernel_query = torch.sum(original_kernel_results_masked, 2) # k*
#        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) #* self.nn_scaler
#        log_per_kernel_query_masked = log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1) # make sure we mask out padding values
#        per_kernel = torch.sum(log_per_kernel_query_masked, 1) 



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

    def forward_representation(self, sequence_embeddings: torch.Tensor, sequence_mask: torch.Tensor, sequence_type:str) -> torch.Tensor:

        if sequence_type == "query":
            positional_features = self.positional_features_q[:,:sequence_embeddings.shape[1],:]
        elif sequence_type == "doc_model":
            positional_features = self.positional_features_d[:,:sequence_embeddings.shape[1],:]
        else: #sequence_type == "doc_pretrain" 
            positional_features = self.positional_features_q if random.random() > 0.5 else self.positional_features_d
            positional_features = positional_features[:,:sequence_embeddings.shape[1],:]

        #sequence_embeddings = sequence_embeddings * sequence_mask.unsqueeze(-1)
        #self.drop(sequence_embeddings)
        if self.use_pos_agnostic == True:
            sequence_embeddings_context = self.contextualizer(sequence_embeddings.transpose(1,0), positional_features.transpose(1,0),src_key_padding_mask=~sequence_mask.bool()).transpose(1,0)
        else:
            sequence_embeddings_context = self.contextualizer(self.layer_norm(sequence_embeddings + positional_features).transpose(1,0),src_key_padding_mask=~sequence_mask.bool()).transpose(1,0)
            #sequence_embeddings_context = self.contextualizer((sequence_embeddings + positional_features).transpose(1,0),src_key_padding_mask=~sequence_mask.bool()).transpose(1,0)
        
        #sequence_embeddings = (self.mixer * self.layer_norm(sequence_embeddings) + (1 - self.mixer) * sequence_embeddings_context) * sequence_mask.unsqueeze(-1)

        return sequence_embeddings_context


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
        "position_bias: "+str(self.position_bias.data) + "position_bias_absolute: "+str(self.position_bias_absolute.data) +"mixer: "+str(self.mixer.data)

    def get_param_secondary(self):
        return {"dense_weight":self.dense.weight,#"dense_bias":self.dense.bias,
                #"dense_mean_weight":self.dense_mean.weight,#"dense_mean_bias":self.dense_mean.bias,
                #"dense_comb_weight":self.dense_comb.weight, 
                #"scaler":self.nn_scaler ,
                "mixer":self.mixer}

from matchmaker.modules.relative_position_transformer import TransformerEncoderLayerRelative, TransformerEncoderRelative

class TK_v6_relative(nn.Module):
    '''
    TK is a neural IR model - a fusion between transformer contextualization & kernel-based scoring

    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions

    '''

    @staticmethod
    def from_config(config,word_embeddings_out_dim):
        return TK_v6_relative(word_embeddings_out_dim, 
                     kernels_mu =    config["tk_kernels_mu"],
                     kernels_sigma = config["tk_kernels_sigma"],
                     att_heads =     config["tk_att_heads"],
                     att_layer =     config["tk_att_layer"],
                     att_proj_dim =  config["tk_att_proj_dim"],
                     att_ff_dim =    config["tk_att_ff_dim"],
                     max_length =    config["max_doc_length"],
                     #use_pos_agnostic     = config["tk_use_pos_agnostic"],
                     use_position_bias    = config["tk_use_position_bias"],  
                     use_diff_posencoding = config["tk_use_diff_posencoding"],
                     position_bias_bin_percent   = config["tk_position_bias_bin_percent"],
                     position_bias_absolute_steps= config["tk_position_bias_absolute_steps"] )

    def __init__(self,
                 _embsize:int,
                 kernels_mu: List[float],
                 kernels_sigma: List[float],
                 att_heads: int,
                 att_layer: int,
                 att_proj_dim: int,
                 att_ff_dim: int,
                 max_length,
                 #use_pos_agnostic,  
                 use_position_bias,
                 use_diff_posencoding,
                 position_bias_bin_percent,
                 position_bias_absolute_steps):

        super(TK_v6_relative, self).__init__()

        n_kernels = len(kernels_mu)
        #self.use_pos_agnostic     = use_pos_agnostic    
        self.use_position_bias    = use_position_bias   
        self.use_diff_posencoding = use_diff_posencoding
        self.position_bias_bin_percent = position_bias_bin_percent
        self.position_bias_absolute_steps = position_bias_absolute_steps

        if len(kernels_mu) != len(kernels_sigma):
            raise Exception("len(kernels_mu) != len(kernels_sigma)")

        # static - kernel size & magnitude variables
        self.mu = Variable(torch.cuda.FloatTensor(kernels_mu), requires_grad=False).view(1, 1, 1, n_kernels)
        self.sigma = Variable(torch.cuda.FloatTensor(kernels_sigma), requires_grad=False).view(1, 1, 1, n_kernels)
        #self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))


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


        self.position_bias = nn.Parameter(torch.full([int(1/self.position_bias_bin_percent) + 1], 1, dtype=torch.float32, requires_grad=True))
        self.position_bias.data[0] = 0 # padding
        #self.position_bias.data[1,:] = 1.3
        #self.position_bias.data[2,:] = 1.2
        #self.position_bias.data[3,:] = 1.1

        self.position_bias_absolute_factors = math.ceil(max_length/self.position_bias_absolute_steps)

        self.position_bias_index_selects = torch.tensor([i + 1 for i in range(self.position_bias_absolute_factors) for _ in range(self.position_bias_absolute_steps)]).float().cuda()
        self.position_bias_absolute = nn.Parameter(torch.full([self.position_bias_absolute_factors + 1], 1, dtype=torch.float32, requires_grad=True))
        self.position_bias_absolute.data[0] = 0 # padding
        #self.position_bias_absolute.data[1:4,:] = 1.3
        #self.position_bias_absolute.data[4:7,:] = 1.2
        #self.position_bias_absolute.data[7:11,:]= 1.1


        encoder_layer = TransformerEncoderLayerRelative(_embsize, att_heads, dim_feedforward=att_ff_dim, dropout=0)
        self.contextualizer = TransformerEncoderRelative(encoder_layer, att_layer, norm=None)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights) 
        self.cosine_module = CosineMatrixAttention()

        self.dense = nn.Linear(n_kernels, 1, bias=False)
        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014) # inits taken from matchzoo

    def clear_explicit_bias(self):
        self.position_bias.data[1:] = 1
        self.position_bias_absolute.data[1:] = 1

    def forward(self, query_embeddings: torch.Tensor, document_embeddings: torch.Tensor,
                query_pad_oov_mask: torch.Tensor, document_pad_oov_mask: torch.Tensor, 
                output_secondary_output: bool = False) -> torch.Tensor:
        # pylint: disable=arguments-differ

        #
        # contextualization
        # -------------------------------------------------------

        query_embeddings = self.forward_representation(query_embeddings, query_pad_oov_mask,self.positional_features_q[:,:query_embeddings.shape[1],:])
        document_embeddings = self.forward_representation(document_embeddings, document_pad_oov_mask,self.positional_features_d[:,:document_embeddings.shape[1],:])

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
        # explicit position bias
        # -------------------------------------------------------

        if self.use_position_bias == True:
    
            mask = torch.zeros_like(document_pad_oov_mask)
            pos_range = (torch.round(doc_lengths*self.position_bias_bin_percent).unsqueeze(-1) * torch.arange(0,self.position_bias.shape[0] - 1,device="cuda").float()).long()
            for b in range(mask.shape[0]):
                mask[b,pos_range[b]] = 1
            cum_sum = torch.cumsum(mask,dim = 1)
    
            pos_indices = (cum_sum * document_pad_oov_mask).long().view(-1)
            position_bias_map = torch.index_select(self.position_bias,0,pos_indices).view(mask.shape[0],1,-1,1)
    
            abs_indices = (self.position_bias_index_selects[:document_embeddings.shape[1]] * document_pad_oov_mask).long().view(-1)
            absolute_bias = torch.index_select(self.position_bias_absolute,0,abs_indices).view(mask.shape[0],1,-1,1)

            kernel_results_masked = kernel_results_masked * position_bias_map * absolute_bias

        #
        # kernel-pooling
        # -------------------------------------------------------

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) #* self.nn_scaler
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

        #if positional_features is None:
        #    positional_features = self.positional_features_d[:,:sequence_embeddings.shape[1],:]

        sequence_embeddings = sequence_embeddings * sequence_mask.unsqueeze(-1)

        #if self.use_pos_agnostic == True:
        #    sequence_embeddings_context = self.contextualizer(sequence_embeddings.transpose(1,0), positional_features.transpose(1,0),src_key_padding_mask=~sequence_mask.bool()).transpose(1,0)
        #else:
        sequence_embeddings_context = self.contextualizer(sequence_embeddings,src_key_padding_mask=~sequence_mask.bool().unsqueeze(1))
        
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
        "position_bias: "+str(self.position_bias.data) + "position_bias_absolute: "+str(self.position_bias_absolute.data) +"mixer: "+str(self.mixer.data)

    def get_param_secondary(self):
        return {"dense_weight":self.dense.weight,#"dense_bias":self.dense.bias,
                #"dense_mean_weight":self.dense_mean.weight,#"dense_mean_bias":self.dense_mean.bias,
                #"dense_comb_weight":self.dense_comb.weight, 
                #"scaler":self.nn_scaler ,
                "mixer":self.mixer}