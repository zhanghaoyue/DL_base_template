import torch
import torch.nn.functional as F
from loss_util import custom_loss
from sklearn.metrics import normalized_mutual_info_score as nmi_score

multiclass = False


def nll_loss(output, target):
    m = torch.nn.LogSoftmax(dim=1)
    return F.nll_loss(m(output), target)


def mse_loss(output, target):
    return F.mse_loss(output, target)


def cross_entropy_loss(output, target):
    weights = [2, 1, 1]
    class_weights = torch.FloatTensor(weights)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    return criterion(output, target.long())


def bce_logits_loss(output, target):
    if target.size() != output.size():
        target = target.unsqueeze(1)
    if multiclass:
        target = target.long()
        target = torch.squeeze(torch.nn.functional.one_hot(target, num_classes=3))
    target = target.float()
    # pos_weight=torch.tensor([6]).cuda() ## add weight to adjust imbalanced data if necessary
    criterion = torch.nn.BCEWithLogitsLoss()
    return criterion(output, target)


def contrastive_loss(output, target):
    output_1 = output[0]
    output_2 = output[1]
    criterion = custom_loss.ContrastiveLoss(margin=2.0)
    return criterion(output_1, output_2, target)


def mutualinfo_loss(output, target):
    output_1 = output[0]
    output_2 = output[1]
    criterion = torch.nn.BCEWithLogitsLoss()
    mutualinfo = nmi_score(output_1.cpu().numpy(), output_2.cpu().numpy())
    return criterion(torch.from_numpy(mutualinfo).float(), target)


def bce_focal_loss(output, target):
    if target.size() != output.size():
        target = target.unsqueeze(1)
    target = target.float()
    alpha = 0.75
    gamma = 1.0
    BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target, reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1 - pt) ** gamma * BCE_loss
    return F_loss.mean()


def multi_focal_loss(output, target):
    alpha = 0.25
    gamma = 2.0
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    ce_loss = criterion(output, target.long())
    pt = torch.exp(-ce_loss)
    focal_loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()
    return focal_loss
