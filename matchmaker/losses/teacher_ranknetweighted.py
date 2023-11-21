import torch.nn as nn
import torch
from torch.nn import BCEWithLogitsLoss


class RankNetTeacher(nn.Module):
    def __init__(self):
        super(RankNetTeacher, self).__init__()

        #self.bce = torch.nn.BCEWithLogitsLoss()


    def forward(self, scores_pos, scores_neg, label_pos, label_neg):
        """
        """                      
        x = scores_pos - scores_neg
        #y = label_pos - label_neg

        # 0-1 normalize the positive labels (and add a small number to avoid zero influence of the lowest score)
        weights = (label_pos - label_pos.min()) / (label_pos.max() - label_pos.min()) + 0.1

        loss = torch.nn.BCEWithLogitsLoss(reduction="sum",weight=weights)(x, torch.ones_like(x))
        return loss