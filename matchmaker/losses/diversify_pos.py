import torch.nn as nn
import torch

class DivQueryLoss(nn.Module):
    def __init__(self):
        super(DivQueryLoss, self).__init__()

    def forward(self, distances_query):
        """

        """
        loss = torch.mean(torch.pow((distances_query),2))
        return loss