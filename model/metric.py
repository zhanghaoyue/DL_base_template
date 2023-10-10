import numpy as np
import sklearn.metrics as skmetrics
import torch


def accuracy(output, target):
    with torch.no_grad():
        output = torch.sigmoid(output)
        target = torch.unsqueeze(target, 1)
        target = target.cpu().detach().numpy()
        if output.shape != target.shape:
            pred = torch.topk(output, 1, dim=1)[1]
            assert pred.shape[0] == len(target)
            pred = pred.cpu().detach().numpy()
            accu = skmetrics.accuracy_score(target, pred)
        else:
            output = output.cpu().detach().numpy()
            threshold = Find_Optimal_Cutoff(output, target)
            accu = skmetrics.accuracy_score(target, (output >= threshold))
    return accu


def precision_score(output, target):
    with torch.no_grad():
        output = torch.sigmoid(output)
        target = torch.unsqueeze(target, 1)
        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        threshold = Find_Optimal_Cutoff(output, target)
        precision = skmetrics.precision_score(target, (output >= threshold))
    return precision


def sensitivity(output, target):
    with torch.no_grad():
        output = torch.sigmoid(output)
        target = torch.unsqueeze(target, 1)
        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        fpr, tpr, threshold = Find_Optimal_Cutoff_2(output, target)
        sens = tpr
    return sens


def specificity(output, target):
    with torch.no_grad():
        output = torch.sigmoid(output)
        target = torch.unsqueeze(target, 1)
        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        fpr, tpr, threshold = Find_Optimal_Cutoff_2(output, target)
        spec = 1 - fpr
    return spec


def sens_spec_combo(output, target):
    with torch.no_grad():
        pred = torch.sigmoid(output)
        truth = torch.unsqueeze(target, 1)
        pred = pred.cpu().detach().numpy()
        truth = truth.cpu().detach().numpy()
        fpr, tpr, threshold = Find_Optimal_Cutoff_2(truth, pred)
        sens = tpr
        spec = 1 - fpr
        combo = (0.8 * sens + 1.2 * spec) / 2
    return combo


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape == len(target)
        target = torch.unsqueeze(target, 1)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def auc(output, target):
    with torch.no_grad():
        output = torch.sigmoid(output)
        target = torch.unsqueeze(target, 1)
        output = output.cpu()
        target = target.cpu()
        try:
            if output.shape != target.shape:
                target = torch.squeeze(torch.nn.functional.one_hot(target, num_classes=3))
                roc_auc = skmetrics.roc_auc_score(target, output, average="weighted", multi_class='ovr')
            else:
                roc_auc = skmetrics.roc_auc_score(target, output)
        except ValueError:
            roc_auc = 0.0
    return roc_auc


def auc_roc_score(output, target):
    """Computes the area under the receiver operator characteristic (ROC) curve using the trapezoid method. \
    Restricted binary classification tasks."""

    fpr, tpr = roc_curve(output[0], target)

    d = fpr[1:] - fpr[:-1]

    sl1, sl2 = [slice(None)], [slice(None)]

    sl1[-1], sl2[-1] = slice(1, None), slice(None, -1)

    return (d * (tpr[tuple(sl1)] + tpr[tuple(sl2)]) / 2.).sum(-1)


def roc_curve(output, target):
    """Computes the receiver operator characteristic (ROC) curve by determining the true positive ratio (TPR) \
    and false positive ratio (FPR) for various classification thresholds. Restricted binary classification tasks."""

    target = (target == 1)

    desc_score_indices = torch.flip(output.argsort(-1), [-1])

    output = output[desc_score_indices]

    target = target[desc_score_indices]

    d = output[1:] - output[:-1]

    distinct_value_indices = torch.nonzero(d).transpose(0, 1)[0]

    threshold_idxs = torch.cat((distinct_value_indices, torch.LongTensor([len(target) - 1]).to(target.device)))

    tps = torch.cumsum(target * 1, dim=-1)[threshold_idxs]

    fps = (1 + threshold_idxs - tps)

    if tps[0] != 0 or fps[0] != 0:
        fps = torch.cat((torch.LongTensor([0]), fps))

        tps = torch.cat((torch.LongTensor([0]), tps))

    fpr, tpr = fps.float() / fps[-1], tps.float() / tps[-1]

    return fpr, tpr


def Find_Optimal_Cutoff(output, target):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    output : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = skmetrics.roc_curve(target, output)

    best_threshold_idx = np.argmax(tpr - fpr)
    best_threshold = threshold[best_threshold_idx]

    return best_threshold


def Find_Optimal_Cutoff_2(output, target):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    output : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = skmetrics.roc_curve(target, output)

    best_threshold_idx = np.argmax(tpr - fpr)
    best_threshold = threshold[best_threshold_idx]
    best_fpr = fpr[best_threshold_idx]
    best_tpr = tpr[best_threshold_idx]
    return best_fpr, best_tpr, best_threshold
