import torch.nn as nn
import torch

class MLMLoss(nn.Module):
    def __init__(self):
        super(MLMLoss, self).__init__()

        self.loss = torch.nn.CrossEntropyLoss() #reduction="mean", ignore_index=-100)

    def forward(self, pred_scores, labels):
        """
        Masked-Language-Modelling Loss
        """
        # pred_scores (bs, seq_length, vocab_size)
        # labels (bs, seq_length)

        loss = self.loss(pred_scores.view(-1, pred_scores.size(-1)), labels["input_ids"].view(-1))
        return loss