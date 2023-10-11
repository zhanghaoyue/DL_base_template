import warnings
import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction


class _Loss(Module):
    def __init__(self, reduction='mean'):
        super(_Loss, self).__init__()
        self.reduction = reduction


class BCEWithCosineLoss(_Loss):
    __constants__ = ['weight', 'pos_weight', 'reduction', 'dim', 'eps']

    def __init__(self, weight=None, reduction='mean', pos_weight=None, dim=1, eps=1e-8):
        super(BCEWithCosineLoss, self).__init__(reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
        self.dim = dim
        self.eps = eps

    def forward(self, input, target):
        BCE = F.binary_cross_entropy_with_logits(input[0], target, self.weight, pos_weight=self.pos_weight,
                                                 reduction=self.reduction)

        cosine = F.cosine_similarity(input[1], input[2], self.dim, self.eps)
        BCEcosine = F.binary_cross_entropy_with_logits(cosine, target, self.weight, pos_weight=self.pos_weight,
                                                       reduction=self.reduction)

        return BCE + BCEcosine


class ContrastiveLoss(nn.Module):
    """

    Contrastive loss

    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise

    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()

        self.margin = margin

        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances

        losses = 0.5 * (target.float() * distances +

                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))

        return losses.mean() if size_average else losses.sum()