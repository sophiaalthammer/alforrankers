import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import T5Tokenizer


class CrossEntropyPointLoss(nn.Module):
    def __init__(self, config):
        super(CrossEntropyPointLoss, self).__init__()

        tokenizer = T5Tokenizer.from_pretrained(config["bert_pretrained_model"])

        self.ce  = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none') #default use: 'mean'

    def forward(self, output, target, ibn=False):
        if not ibn:
            output_logits = output.logits.view(-1, output.logits.shape[-1])
        else:
            output_logits = output.view(-1, output.shape[-1])
        target = target.view(-1)
        loss = self.ce(output_logits, target)
        return loss

#
# class CrossEntropyPNLoss(nn.Module):
#     def __init__(self, config):
#         super(CrossEntropyPNLoss, self).__init__()
#
#         tokenizer = T5Tokenizer.from_pretrained(config["bert_pretrained_model"])
#
#         #self.ce_pn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
#         self.nll = nn.NLLLoss(ignore_index=tokenizer.pad_token_id)
#
#     def forward(self, output_pos, output_neg, target):
#         print('logits and softmax to the last dimension')
#         print(output_pos.logits)
#         print(F.softmax(output_pos.logits, dim=-1))
#         x = F.softmax(output_pos.logits, dim=-1) - F.softmax(output_neg.logits, dim=-1)
#
#         # it happens that there are negative terms in the output, beacuse the probability for outputneg is higher...
#         # i dont think it works like that!
#
#         print('pos and neg together')
#         print(x)
#
#         output_softmax = torch.square(x.view(-1, output_pos.logits.shape[-1]))
#
#         print(output_softmax)
#
#         target = target.view(-1)
#         loss = self.nll(torch.log(output_softmax), target)
#         return loss