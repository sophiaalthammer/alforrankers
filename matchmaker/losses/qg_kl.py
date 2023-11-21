import torch
import torch.nn as nn
import torch.nn.functional as F

class KLDiv(nn.Module):
    def __init__(self):
        super(KLDiv, self).__init__()

        self.kl  = nn.KLDivLoss(reduction="batchmean")

    def forward(self, output, targets, ibn=False):
        if not ibn:
            output_logits = output.logits.view(-1, output.logits.shape[-1])
        else:
            output_logits = output.view(-1, output.shape[-1])
        log_output = F.log_softmax(output_logits, dim=0) #which dimension here is the right one?
        targets = targets.view(-1, output.logits.shape[-1])  #or without?
        loss = self.kl(log_output, targets)
        return loss

    # from claps
    #  kl = kl_crit(perturb_log_probs.view(-1, vocab_size), true_probs.view(-1, vocab_size))
    #  kl = kl / torch.sum(dec_mask).float()

