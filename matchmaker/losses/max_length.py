import torch.nn as nn
import torch

class MaxLengthLoss(nn.Module):
    def __init__(self):
        super(MaxLengthLoss, self).__init__()

    def forward(self, distances_query):
        """

        """
        loss = -1 * torch.mean(distances_query) #torch.mul( ,-1) ?

        return loss