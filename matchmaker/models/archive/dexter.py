from typing import Dict, Union
import torch
from torch import nn
from torch.nn import functional as F

from transformers import AutoModel
from matchmaker.models.bert_dot import BERT_Dot

class Dexter(BERT_Dot):
    """
    this model does not concat query and document, rather it encodes them sep. and uses a dot-product between the two cls vectors
    """
    def __init__(self,
                 bert_model: Union[str, AutoModel],
                 dropout: float = 0.0,
                 return_vecs: bool = False,
                 codebook_size: int = 2000, 
                 use_gated_sum: bool = False,
                 hard_probability: float = 1.,
                 tau: float = 1.,
                 pooling: str = 'mean',
                 num_samples: int = 1,

                 trainable: bool = True) -> None:
        super().__init__(bert_model,dropout,trainable)

        self.return_vecs = return_vecs

        self.indexer = GumbelIndexer(
            self.bert_model.config.hidden_size, 
            codebook_size,
            use_gated_sum=use_gated_sum,
            hard_probability=hard_probability,
            tau=tau,
        )
        self.conf_num_samples = num_samples
        self.pooling = pooling

    @property
    def num_samples(self):
        if self.training:
            return self.conf_num_samples
        return 1

    def forward(self,
                query: Dict[str, torch.LongTensor],
                document: Dict[str, torch.LongTensor],
                use_fp16:bool = True,
                output_secondary_output: bool = False) -> Dict[str, torch.Tensor]:
        
        with torch.cuda.amp.autocast(enabled=use_fp16):

            query_vecs = self.forward_representation(query)
            document_vecs = self.forward_representation(document)

            query_vecs = query_vecs.flatten(0, -2)
            document_vecs = document_vecs.flatten(0, -2)


            score = torch.bmm(query_vecs.unsqueeze(dim=1), document_vecs.unsqueeze(dim=2)).squeeze(-1).squeeze(-1)

            # used for in-batch negatives, we return them for multi-gpu sync -> out of the forward() method
            if self.training and self.return_vecs:
                score = (score, query_vecs, document_vecs)

            if output_secondary_output:
                return score, {}
            return score

    def forward_representation(self,  # type: ignore
                               tokens: Dict[str, torch.LongTensor],
                               sequence_type="n/a"):

        vecs = self.bert_model(input_ids=tokens["tokens"]["token_ids"],
                                     attention_mask=tokens["tokens"]["mask"])[0]

        #outputs: BaseModelOutputWithPooling = super().forward(*args, **kwargs, return_dict=True)
        h, indices = self.indexer(vecs, num_samples=self.num_samples)
        
        if h.dim() > 3:
            h = h.mean(-2) # Average over num gumbel samples
        
        # Pool to "latent-BOW" somehow (but don't avg over samples yet)
        if self.pooling == 'sum':
            indices = indices.sum(dim=1)
            h = h.sum(dim=1)
        elif self.pooling == 'max':
            indices = indices.max(dim=1).values
            h = h.max(dim=1).values
        elif self.pooling == 'mean':
            indices = indices.mean(dim=1)
            h = h.mean(dim=1)

        #o = dict(outputs)
        #o['last_hidden_state'] = h
        #outputs = DexterModelOutput(**o, indices=indices)
        return h

    def get_param_stats(self):
        return "Dexter: / "
    def get_param_secondary(self):
        return {}



#class DexterConfig(BertConfig):
#    def __init__(
#        self, 
#        codebook_size: int = 2000, 
#        use_gated_sum: bool = True,
#        hard_probability: float = 1.,
#        tau: float = 1.,
#        num_samples: int = 1,
#        **kwargs
#    ):
#        super().__init__(**kwargs)
#        self.codebook_size = codebook_size
#        self.use_gated_sum = use_gated_sum
#        self.hard_probability = hard_probability
#        self.tau = tau
#        self.num_samples = num_samples

       

#class DexterModel(BertModel):
#    """
#    You should be able to change what this subclasses to try out different models.
#    By default, you can use like:
#    
#    ```
#    config = DexterConfig.from_pretrained(
#      'bert-base-uncased', 
#      codebook_size=codebook_size, 
#      use_gated_sum=use_gated_sum,
#      hard_probability=hard_probability,
#      tau=tau,
#      num_samples=num_samples
#    )
#    model = DexterModel.from_pretrained('bert-base-uncased', config=config).to(device)
#    
#    outputs = model(**model_inputs)
#    print(outputs.last_hidden_state.size())
#    print(outputs.indices.size())
#    
#    # you should also be able to save it like any other HF module, but haven't tried yet
#    model.save_pretrained('...')
#    ```
#    """
#    def __init__(self, *args, **kwargs):
#        super().__init__(*args, **kwargs)
#        # freeze(self)
#        self.indexer = GumbelIndexer(
#            self.config.hidden_size, 
#            self.config.codebook_size,
#            use_gated_sum=self.config.use_gated_sum,
#            hard_probability=self.config.hard_probability,
#            tau=self.config.tau,
#        )
#
#    def forward(self, *args, return_dict: bool = None, pooling: str = 'sum', **kwargs):
#        """
#        Returns standard BERT-style output, but: 
#        * last_hidden_state: Batch x Words x D
#          * Embeddings from the indexer (one of Codebook_Size <<< |Vocab|)
#        * indices: Batch x Words x Num_Samples x Codebook_Size, one-hot along codebook dim
#          * NOTE: Pooling this to a final "latent-BOW" seems tricky. I've just been doing 
#            `indices.sum(1)` or `indices.max(1).values`.
#        """
#        outputs: BaseModelOutputWithPooling = super().forward(*args, **kwargs, return_dict=True)
#        h, indices = self.indexer(outputs.last_hidden_state, num_samples=self.num_samples)
#        
#        if h.dim() > 3:
#            h = h.mean(-2) # Average over num gumbel samples
#        
#        # Pool to "latent-BOW" somehow (but don't avg over samples yet)
#        if pooling == 'sum':
#            indices = indices.sum(dim=1)
#        elif pooling == 'max':
#            indices = indices.max(dim=1).values
#        elif pooling == 'mean':
#            indices = indices.mean(dim=1)
#        o = dict(outputs)
#        o['last_hidden_state'] = h
#        outputs = DexterModelOutput(**o, indices=indices)
#        return outputs
#
#    @property
#    def num_samples(self):
#        if self.training:
#            return self.config.num_samples
#        return 1


# This is verbatim from allennlp, so use their instead if you have it installed.
class GatedSum(torch.nn.Module):
    # https://github.com/allenai/allennlp/blob/master/allennlp/modules/gated_sum.py
    def __init__(self, input_dim: int, activation: nn.Module = nn.Sigmoid()) -> None:
        super().__init__()
        self.input_dim = input_dim
        self._gate = torch.nn.Linear(input_dim * 2, 1)
        self._activation = activation

    def get_input_dim(self):
        return self.input_dim

    def get_output_dim(self):
        return self.input_dim

    def forward(self, input_a: torch.Tensor, input_b: torch.Tensor) -> torch.Tensor:
        if input_a.size() != input_b.size():
            raise ValueError("The input must have the same size.")
        if input_a.size(-1) != self.input_dim:
            raise ValueError("Input size must match `input_dim`.")
        gate_value = self._activation(self._gate(torch.cat([input_a, input_b], -1)))
        return gate_value * input_a + (1 - gate_value) * input_b


class GumbelSoftmax(nn.Module):
    def __init__(
        self,
        tau: float = 1.,
        hard_p: float = 1.,
    ):
        super().__init__()
        
        self._hard_p = hard_p
        self._tau = tau

    def _hard(self):
        """Randomly use continuous relaxation instead of fully discrete, hp% of the time (defaults to never)."""
        return (torch.rand(1).item() < self._hard_p)
      
    def forward(self, x, hard: bool = True, **kwargs):
        """Gumbel during training, exact during inference."""
        if self.training:
            return F.gumbel_softmax(x, tau=self._tau, hard=hard and self._hard(), **kwargs)
        D = x.size(-1)
        return F.one_hot(x.softmax(dim=-1).argmax(-1), num_classes=D).float()

class GumbelIndexer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        pre_activation: nn.Module = nn.Identity(),
        output_dim: int = None,
        use_gated_sum: bool = False,
        hard_probability: float = 1.,
        tau: float = 1.,
        dropout: float = 0.
    ):
        super().__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim or input_dim

        self._codebook_size = codebook_size
        self._hard_p = hard_probability
        
        self._embeddings = nn.Embedding(self._codebook_size, self._output_dim).weight
        self._codebook = nn.Sequential(
            pre_activation,
            nn.Dropout(dropout),
            nn.Linear(self._output_dim, self._codebook_size, bias=False) # not sure if bias would make difference here
        )
        
        self._tau = tau
        self._activation = GumbelSoftmax(tau=tau, hard_p=hard_probability)

        self._dense_gate = use_gated_sum and GatedSum(input_dim)

    def forward(
        self, 
        x: torch.Tensor,
        num_samples: int = 1
    ):
        logits = self._codebook(x)
        logits = logits.unsqueeze(-2).expand(-1, -1, num_samples, -1) # B x W x S x CS

        indices = self._activation(logits.type(torch.float32), hard=True) # Avoid FP16 problems
        assert not torch.isnan(indices).any(), "Indices NaN."

        embeddings = (indices @ self._embeddings).type_as(x)

        if self._dense_gate:
            embeddings = self._dense_gate(x.unsqueeze(-2).expand_as(embeddings), embeddings)

        return embeddings, indices