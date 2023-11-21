from typing import Dict, Union
import torch
from transformers import DistilBertConfig

from transformers import AutoModel
from matchmaker.models.bert_cat import BERT_Cat,BERT_Cat_Config
#from transformers import Transformer
import math

class Bert_dot_qa(BERT_Cat):
    """
    this model does not concat query and document, rather it encodes them sep. and uses a dot-product between the two cls vectors
    """
    def __init__(self,
                 bert_model: Union[str, AutoModel],
                 dropout: float = 0.0,
                 return_vecs: bool = False,
                 rcr_main_compress_dim = -1,
                 rcr_residual_compress_dim= -1,
                 trainable: bool = True) -> None:
        cfg = BERT_Cat_Config()
        cfg.bert_model          = bert_model
        cfg.trainable           = trainable
        super().__init__(cfg)

        self._classification_layer = None
        del self._classification_layer
        self.return_vecs = return_vecs

        self.use_main_compression = rcr_main_compress_dim > -1
        if self.use_main_compression:
            self.compressor_retrieval = torch.nn.Linear(self.bert_model.config.hidden_size, rcr_main_compress_dim)
            #self.compressor_refine_query = torch.nn.Linear(self.bert_model.config.hidden_size, 384)

        self.use_residual_compression = rcr_residual_compress_dim > -1
        if self.use_residual_compression:
            self.compressor_refinement = torch.nn.Linear(self.bert_model.config.hidden_size, rcr_residual_compress_dim)
            self.de_compressor_refinement = torch.nn.Linear(rcr_residual_compress_dim, rcr_main_compress_dim)

            #self.de_compressor_helper = torch.nn.Linear(rcr_main_compress_dim, self.bert_model.config.hidden_size)

            #self.compression_scaler = self.bert_model.config.hidden_size // rcr_main_compress_dim
            # merge option
            #total_out_size = self.compressor_retrieval.out_features + self.compressor_refinement.out_features
            #self.merge_refinement = torch.nn.Linear(total_out_size, total_out_size)
        
        self.use_bitwise_binarization = False


        #torch.nn.init.constant_(self.one_hot_reduction.weight, 0.001)
        #torch.nn.init.constant_(self.one_hot_reduction.bias, 1)

        #self.score_merge = torch.nn.Linear(2, 1, bias=False)
        #torch.nn.init.constant_(self.score_merge.weight, 1)

        #self.quant = torch.quantization.QuantStub()
        #self.dequant = torch.quantization.DeQuantStub()
        #encoder_layer = torch.nn.TransformerEncoderLayer(32, 8, dim_feedforward=64, dropout=0)
        #cfg = DistilBertConfig()
        #cfg.n_layers = 1
        #cfg.dim = 64
        #cfg.n_heads = 8
        #cfg.dropout = 0
        #cfg.hidden_dim = 128
        #self.contextualizer_refinement = Transformer(cfg)
        #self.contextualizer_refinement_scorer = torch.nn.Linear(64, 1)
        #torch.nn.init.constant_(self.contextualizer_refinement_scorer.bias, 1)



    #def reanimate(self,added_bias,layers):
    #    self.bert_model.reanimate(added_bias,layers)

    def forward(self,
                query: Dict[str, torch.LongTensor],
                document: Dict[str, torch.LongTensor],
                global_step = -1,
                use_fp16:bool = True,
                output_secondary_output: bool = False) -> Dict[str, torch.Tensor]:
        
        with torch.cuda.amp.autocast(enabled=use_fp16):
           
            query_vecs = self.forward_representation(query)
            document_vecs = self.forward_representation(document)

            #
            # retrieval score
            #
            if self.use_main_compression:
                q_ret = self.compressor_retrieval(query_vecs[:,0,:])
                d_ret = self.compressor_retrieval(document_vecs[:,0,:])
            #score_retrieval = torch.bmm(q_ret.unsqueeze(dim=1),
            #                            d_ret.unsqueeze(dim=2)).squeeze(-1)#.squeeze(-1)
            else:
                q_ret = query_vecs[:,0,:]
                d_ret = document_vecs[:,0,:]


            #
            # refinement score
            #
            #q_r = self.compressor_refinement(query_vecs)
            #d_r = self.compressor_refinement(document_vecs)

            # v1 transformer idea
            #merged = torch.cat([q_r, d_r],dim=1)#.transpose(1,0)
            #merged_mask = torch.cat([query["attention_mask"], document["attention_mask"]],dim=1)
#
            #tf_refine = self.contextualizer_refinement(merged,merged_mask,head_mask = [None] )# *1)#.transpose(1,0)
#
            #score_refine = self.contextualizer_refinement_scorer(tf_refine[0][:,0,:]) * 50 

            # v2 residual merge idea
            if self.use_residual_compression:
                q_r = self.compressor_refinement(query_vecs)
                d_r = self.compressor_refinement(document_vecs)

                if self.use_bitwise_binarization:
                    if self.training and global_step > -1:
                        scale = math.pow((1.0 + global_step * 0.1), 0.5)
                        
                        q_r = torch.tanh(q_r * scale)
                        d_r = torch.tanh(d_r * scale)

                    else: # hard binarization
                        q_r = q_r.new_ones(q_r.size()).masked_fill_(q_r < 0, -1.0)
                        d_r = d_r.new_ones(d_r.size()).masked_fill_(d_r < 0, -1.0)

                q_r = self.de_compressor_refinement(q_r)
                d_r = self.de_compressor_refinement(d_r)

                q_r = q_r + q_ret.unsqueeze(1)
                d_r = d_r + d_ret.unsqueeze(1)

                #compression_error = torch.mean(torch.pow(self.de_compressor_helper(q_r) - query_vecs.detach(),2)[query["attention_mask"].bool()]) + \
                #                    torch.mean(torch.pow(self.de_compressor_helper(d_r) - document_vecs.detach(),2)[document["attention_mask"].bool()])

            else:
                q_r = query_vecs    + q_ret.unsqueeze(1)
                d_r = document_vecs + d_ret.unsqueeze(1)

            #q_r = self.compressor_refinement(query_vecs)
            #d_r = self.compressor_refinement(document_vecs)

            #q_r = torch.cat([q_r, q_ret.unsqueeze(1).expand(-1,q_r.shape[1],-1)],dim=-1)
            #d_r = torch.cat([d_r, d_ret.unsqueeze(1).expand(-1,d_r.shape[1],-1)],dim=-1)
            #q_r = self.quant(q_r)
            #d_r = self.quant(d_r)

            #if self.use_residual_compression:
            #    q_r = self.merge_refinement(q_r)
            #    d_r = self.merge_refinement(d_r)

            score_refine_per_term = torch.bmm(q_r,d_r.transpose(2,1))
            score_refine_per_term[~(document["attention_mask"].bool()).unsqueeze(1).expand(-1, score_refine_per_term.shape[1],-1)] = - 1000

            #score_refine_teacher = torch.bmm(query_vecs,document_vecs.transpose(2,1))
            #score_refine_teacher[~document["attention_mask"].unsqueeze(1).expand(-1, score_refine.shape[1],-1)] = - 1000


            # v2b sparsity on the compression diff
            #compression_error = torch.mean(torch.pow((score_refine) - (score_refine_teacher),2))

            #if self.use_residual_compression:
            #    score_refine_per_term = score_refine_per_term * self.compression_scaler

#
            score_refine = score_refine_per_term.max(-1).values
#
            score_refine[~(query["attention_mask"].bool())] = 0
#
            score_refine = score_refine.sum(-1,keepdim=True)

            #score_refine = self.dequant(score_refine)


            #if self.training:
            #    score = score_retrieval + score_refine
            #else:
            #    score = score_refine
            score = score_refine.squeeze(-1)


            #score = self.score_merge(torch.cat([score_retrieval,score_refine],dim=-1)).squeeze(-1)

            # used for in-batch negatives, we return them for multi-gpu sync -> out of the forward() method
            if self.training and self.return_vecs:
                score = (score, score_refine_per_term)

            if output_secondary_output:
                return score,{}
            return score # ,[compression_error.unsqueeze(0)]

    def forward_representation(self,  # type: ignore
                               tokens: Dict[str, torch.LongTensor],
                               sequence_type=None) -> Dict[str, torch.Tensor]:
        
        if self.bert_model.base_model_prefix == 'distilbert': # diff input / output 
            vecs = self.bert_model(input_ids=tokens["input_ids"],
                                     attention_mask=tokens["attention_mask"])[0]
        elif self.bert_model.base_model_prefix == 'longformer':
            vecs, _ = self.bert_model(input_ids=tokens["input_ids"],
                                        attention_mask=tokens["attention_mask"].long(),
                                        global_attention_mask = ((1-tokens["tokens"]["type_ids"])*tokens["attention_mask"]).long())
        elif self.bert_model.base_model_prefix == 'roberta': # no token type ids
            vecs, _ = self.bert_model(input_ids=tokens["input_ids"],
                                        attention_mask=tokens["attention_mask"])
        elif self.bert_model.base_model_prefix == 'electra':
            vecs = self.bert_model(input_ids=tokens["input_ids"],
                                     token_type_ids=tokens["tokens"]["type_ids"],
                                     attention_mask=tokens["attention_mask"])[0]
        else:
            vecs, _ = self.bert_model(input_ids=tokens["input_ids"],
                                        token_type_ids=tokens["tokens"]["type_ids"],
                                        attention_mask=tokens["attention_mask"])

        #vecs = self.compressor(vecs)

        # assume doc only pre-train a.t.m. 
        if sequence_type == "pretrain":
            d_ret = self.compressor_retrieval(vecs[:,0,:])
            d_r = self.compressor_refinement(vecs)
            d_r = self.merge_refinement(torch.cat([d_r,d_ret.unsqueeze(1).expand(-1,d_r.shape[1],-1)],dim=-1))#).transpose(2,1)
            return d_r

        if sequence_type == "doc_encode" or sequence_type == "query_encode":
            vecs = vecs * tokens["attention_mask"].unsqueeze(-1)

        return vecs

    def get_param_stats(self):
        return "BERT_dot_qa:" #  self.score_merge.weight: "+str(self.score_merge.weight)
    def get_param_secondary(self):
        return {}