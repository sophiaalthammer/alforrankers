import torch.nn as nn
import torch
from torch.nn import BCEWithLogitsLoss


class WeightedHingeLoss(nn.Module):
    def __init__(self):
        super(WeightedHingeLoss, self).__init__()

        self.loss_fn = torch.nn.MarginRankingLoss(margin=1, reduction='none')


    def forward(self, scores_pos, scores_neg, label_pos, label_neg):
        """
        """                      
        #x = scores_pos - scores_neg
        #y = label_pos - label_neg

        # 0-1 normalize the positive labels (and add a small number to avoid zero influence of the lowest score)
        weights = (label_pos - label_pos.min()) / (label_pos.max() - label_pos.min()) + 0.1

        loss = self.loss_fn(scores_pos,scores_neg,torch.ones_like(scores_pos))
        loss = (loss * weights).mean()
        return loss