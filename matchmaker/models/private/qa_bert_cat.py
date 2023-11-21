from typing import Dict, Union,Optional

import torch
from torch import nn as nn

from transformers import AutoModel
from torch.nn import functional as F


class QA_Bert_cat(nn.Module):
    """
    Huggingface LM (bert,distillbert,roberta,albert) model for concated q-d cls scoring 
    """
    def __init__(self,
                 bert_model: Union[str, AutoModel],
                 dropout: float = 0.0,
                 trainable: bool = True) -> None:
        super().__init__()

        if isinstance(bert_model, str):
            self.bert_model = AutoModel.from_pretrained(bert_model)
        else:
            self.bert_model = bert_model

        for p in self.bert_model.parameters():
            p.requires_grad = trainable

        self._dropout = torch.nn.Dropout(p=dropout)

        self._classification_layer = torch.nn.Linear(self.bert_model.config.hidden_size, 1)
        self.start_logits_pooler = PoolerStartLogits(self.bert_model.config.hidden_size)
        self.end_logits_pooler = PoolerEndLogits(self.bert_model.config.hidden_size)
        self.answerability = PoolerAnswerClass(self.bert_model.config.hidden_size)

        self.start_n_top = 4
        self.end_n_top = 4

    def forward(self, tokens: Dict[str, torch.LongTensor],qa_start_labels:torch.Tensor=None,
                use_fp16:bool = True, output_secondary_output: bool = False):

        with torch.cuda.amp.autocast(enabled=use_fp16):
            all_vecs = self.forward_representation(tokens)
            all_vecs = self._dropout(all_vecs)
            score = self._classification_layer(all_vecs[:,0,:]).squeeze(-1)

            qa_mask = (~(tokens["tokens"]["type_ids"].bool())).to(dtype=score.dtype)
            #sep_index = (qa_mask*tokens["attention_mask"]).sum(-1).long().unsqueeze(-1)

            #sep_vectors = all_vecs.gather(-2, sep_index.long().unsqueeze(-1).unsqueeze(-1).expand(-1,-1,all_vecs.shape[-1]))

            answerability = self.answerability(all_vecs[:,0,:])

            qa_logits_start = self.start_logits_pooler(all_vecs,qa_mask)

            slen, hsz = all_vecs.shape[-2:]
            if qa_start_labels!=None: # only during training
                qa_logits_end = []
                #answerability = torch.zeros(1,device=qa_logits_start.device)
                for i in range(len(qa_start_labels[0])):
                    start_positions = qa_start_labels[:,i,None,None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
                    start_positions = start_positions * (start_positions != -1).long()
                    start_states = all_vecs.gather(-2, start_positions)  # shape (bsz, 1, hsz)
                    qa_logits_end.append(self.end_logits_pooler(all_vecs,start_states.expand(-1, slen, -1),p_mask=qa_mask).unsqueeze(1))

                    #answerability = answerability + self.answerability(start_states.squeeze(1),all_vecs[:,0,:])

                #answerability = answerability / len(qa_start_labels[0])
                qa_logits_end = torch.cat(qa_logits_end,dim=1)
            else:
                # todo beam search
                start_states = all_vecs.gather(-2,torch.max(qa_logits_start,dim=-1,keepdim=True).indices.unsqueeze(-1).expand(-1, -1, hsz))

                qa_logits_end = self.end_logits_pooler(all_vecs,start_states.expand(-1, slen, -1),p_mask=qa_mask)

                
                # during inference, compute the end logits based on beam search
                #bsz, slen, hsz = all_vecs.size()
                #start_log_probs = F.softmax(qa_logits_start, dim=-1)  # shape (bsz, slen)
#
                #start_top_log_probs, start_top_index = torch.topk(
                #    start_log_probs, self.start_n_top, dim=-1
                #)  # shape (bsz, start_n_top)
                #start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
                #start_states = torch.gather(all_vecs, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
                #start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)
#
                #hidden_states_expanded = all_vecs.unsqueeze(2).expand_as(
                #    start_states
                #)  # shape (bsz, slen, start_n_top, hsz)
                #p_mask = qa_mask.unsqueeze(-1) if qa_mask is not None else None
                #end_logits = self.end_logits_pooler(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
                #end_log_probs = F.softmax(end_logits, dim=1)  # shape (bsz, slen, start_n_top)
#
                #end_top_log_probs, end_top_index = torch.topk(
                #    end_log_probs, self.end_n_top, dim=1
                #)  # shape (bsz, end_n_top, start_n_top)
                #end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
                #end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

                #start_states = torch.einsum("blh,bl->bh", hidden_states, start_log_probs)
                #cls_logits = self.answer_class(hidden_states, start_states=start_states, cls_index=cls_index)


            #score = score * answerability.softmax(dim=-1)[:,1] # dim = 1 is has_qa_answer

            if output_secondary_output:
                return score, {} ,answerability,qa_logits_start,qa_logits_end
            return score,answerability,qa_logits_start,qa_logits_end

    def forward_representation(self,  # type: ignore
                               tokens: Dict[str, torch.LongTensor],sequence_type="n/a") -> Dict[str, torch.Tensor]:
        
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

        return vecs

    def get_param_stats(self):
        return "bert_cat: / "
    def get_param_secondary(self):
        return {}

# from huggingface
class PoolerAnswerClass(nn.Module):
    """
    Compute SQuAD 2.0 answer class from classification and start tokens hidden states.

    Args:
        config (:class:`~transformers.PretrainedConfig`):
            The config used by the model, will be used to grab the :obj:`hidden_size` of the model.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size // 2)
        self.activation = nn.Tanh()
        self.dense_1 = nn.Linear(hidden_size // 2, 2, bias=False)

    def forward(
        self,
        #start_states: Optional[torch.FloatTensor] = None,
        cls_token_state: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            start_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`, `optional`):
                The hidden states of the first tokens for the labeled span.
            start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                The position of the first token for the labeled span.
            cls_index (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                Position of the CLS token for each sentence in the batch. If :obj:`None`, takes the last token.

        .. note::

            One of ``start_states`` or ``start_positions`` should be not obj:`None`. If both are set,
            ``start_positions`` overrides ``start_states``.

        Returns:
            :obj:`torch.FloatTensor`: The SQuAD 2.0 answer class.
        """
        # No dependency on end_feature so that we can obtain one single `cls_logits` for each sample.


        #x = self.dense_0(torch.cat([start_states, cls_token_state], dim=-1))
        x = self.dense_0(cls_token_state)
        #x = self.activation(x)
        x = self.dense_1(x).squeeze(-1)

        return x

class PoolerStartLogits(nn.Module):
    """
    Compute SQuAD start logits from sequence hidden states.

    Args:
        config (:class:`~transformers.PretrainedConfig`):
            The config used by the model, will be used to grab the :obj:`hidden_size` of the model.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.dense_0 = nn.Linear(hidden_size, hidden_size //2)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size //2)
        self.dense_1 = nn.Linear(hidden_size//2, 1)

    def forward(
        self, hidden_states: torch.FloatTensor, p_mask: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            p_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len)`, `optional`):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        Returns:
            :obj:`torch.FloatTensor`: The start logits for SQuAD.
        """
        x = self.dense_0(hidden_states)
        #x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)

        if p_mask is not None:
            if x.dtype == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x


class PoolerEndLogits(nn.Module):
    """
    Compute SQuAD end logits from sequence hidden states.

    Args:
        config (:class:`~transformers.PretrainedConfig`):
            The config used by the model, will be used to grab the :obj:`hidden_size` of the model and the
            :obj:`layer_norm_eps` to use.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.dense_0 = nn.Linear(hidden_size * 2, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dense_1 = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        start_states: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        p_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            start_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len, hidden_size)`, `optional`):
                The hidden states of the first tokens for the labeled span.
            start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
                The position of the first token for the labeled span.
            p_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len)`, `optional`):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        .. note::

            One of ``start_states`` or ``start_positions`` should be not obj:`None`. If both are set,
            ``start_positions`` overrides ``start_states``.

        Returns:
            :obj:`torch.FloatTensor`: The end logits for SQuAD.
        """
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions)  # shape (bsz, 1, hsz)
            start_states = start_states.expand(-1, slen, -1)  # shape (bsz, slen, hsz)

        x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
        #x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)

        if p_mask is not None:
            if x.dtype == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x