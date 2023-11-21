# could implement a loss with KD by weighting the loss of the sample with the KD scores?
# dynamic_sampler_type == pseudo_label and dynamic query file with scores and ids!
# otherwise: dynamic_sampler: False

import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import T5Tokenizer


class CrossEntropyScoresLoss(nn.Module):
    def __init__(self, config):
        super(CrossEntropyScoresLoss, self).__init__()

        tokenizer = T5Tokenizer.from_pretrained(config["bert_pretrained_model"])

        self.ce  = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')

    def forward(self, output, scores, target):
        output_logits = output.logits.view(-1, output.logits.shape[-1])
        target = target.view(-1)
        loss = self.ce(output_logits, target)
        #print(loss.shape)
        #print(scores.shape)
        scores = scores.repeat(int(int(loss.shape[0])/int(scores.shape[0])))
        #print(scores.shape)
        loss = torch.mul(loss.float(), torch.add(scores.float(),20)) # plus 20 so that the scores are not negative, so needs to be changed according to the scores!
        #print(loss)
        #print(loss.shape)
        return loss


# class CrossEntropyScoresPNLoss(nn.Module):
#     # these losses maybe dont make any sense!
#     def __init__(self, config):
#         super(CrossEntropyScoresPNLoss, self).__init__()
#
#         tokenizer = T5Tokenizer.from_pretrained(config["bert_pretrained_model"])
#
#         #self.ce  = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
#         self.nll = nn.NLLLoss(ignore_index=tokenizer.pad_token_id)
#
#     def forward(self, output_pos, output_neg, scores_pos, scores_neg, target, ib_select=None):
#         x = F.softmax(output_pos.logits, dim=-1) - F.softmax(output_neg.logits, dim=-1)
#         output_softmax = x.view(-1, output_pos.logits.shape[-1])
#
#         # it happens that there are negative terms in the output, beacuse the probability for outputneg is higher...
#         # i dont think it works like that!
#
#         target = target.view(-1)
#         loss = self.nll(torch.log(output_softmax), target)
#
#         scores = scores_pos - scores_neg
#         scores = scores.repeat(int(int(loss.shape[0])/int(scores.shape[0])))
#
#         loss = torch.mul(loss.float(), torch.add(scores.float(), 20))
#         return loss