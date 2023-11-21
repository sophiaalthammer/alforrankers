from typing import Dict, Union
import torch

from transformers import AutoModel
from matchmaker.models.bert_cat import BERT_Cat

class Bert_dot_onehot(BERT_Cat):
    """
    this model does not concat query and document, rather it encodes them sep. and uses a dot-product between the two cls vectors
    """
    def __init__(self,
                 bert_model: Union[str, AutoModel],
                 dropout: float = 0.0,
                 return_vecs: bool = False,
                 trainable: bool = True) -> None:
        super().__init__(bert_model,dropout,trainable)

        self._classification_layer = None
        del self._classification_layer
        self.return_vecs = return_vecs

        self.one_hot_reduction = torch.nn.Linear(self.bert_model.config.vocab_size, 128)
        #torch.nn.init.constant_(self.one_hot_reduction.weight, 0.001)
        #torch.nn.init.constant_(self.one_hot_reduction.bias, 1)

        #self.score_merge = torch.nn.Linear(2, 1, bias=False)
        #torch.nn.init.constant_(self.score_merge.weight, 1)

    def reanimate(self,added_bias,layers):
        self.bert_model.reanimate(added_bias,layers)

    def forward(self,
                query: Dict[str, torch.LongTensor],
                document: Dict[str, torch.LongTensor],
                use_fp16:bool = True,
                output_secondary_output: bool = False) -> Dict[str, torch.Tensor]:
        
        with torch.cuda.amp.autocast(enabled=use_fp16):

            query_onehot = torch.nn.functional.one_hot(query["tokens"]["token_ids"],self.one_hot_reduction.in_features).sum(1).float()
            document_onehot = torch.nn.functional.one_hot(document["tokens"]["token_ids"],self.one_hot_reduction.in_features).sum(1).float()

            query_onehot = self.one_hot_reduction(query_onehot)
            document_onehot = self.one_hot_reduction(document_onehot)
            
            query_vecs = self.forward_representation(query)
            document_vecs = self.forward_representation(document)

            query_vecs = torch.cat([query_vecs,query_onehot],dim=1)
            document_vecs = torch.cat([document_vecs,document_onehot],dim=1)

            score = torch.bmm(query_vecs.unsqueeze(dim=1), document_vecs.unsqueeze(dim=2)).squeeze(-1).squeeze(-1)

            # used for in-batch negatives, we return them for multi-gpu sync -> out of the forward() method
            if self.training and self.return_vecs:
                score = (score, query_vecs, document_vecs)

            if output_secondary_output:
                return score,{}
            return score

    def get_param_stats(self):
        return "BERT_dot_onehot: / "
    def get_param_secondary(self):
        return {}