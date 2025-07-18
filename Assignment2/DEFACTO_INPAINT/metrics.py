import torch

# 二枚の白黒画像の一致度を調べる
def region_consistency_metric(estimated, gt, threshold=0.2):

    def ratio(a, b):
        indices = b.nonzero()
        if len(indices) == 0:
            return 0
        else:
            return float(torch.mean(a[indices] / b[indices]))

    one = torch.ones(gt.size()).to(gt.device)
    zero = torch.zeros(gt.size()).to(gt.device)
    gt = torch.where(gt > threshold, one, zero)
    estimated = torch.where(estimated > threshold, one, zero)
    intersection = estimated * gt
    union = estimated + gt
    union = torch.where(union > 1, one, union)
    E = torch.sum(estimated, dim=(1, 2, 3))
    G = torch.sum(gt, dim=(1, 2, 3))
    I = torch.sum(intersection, dim=(1, 2, 3))
    U = torch.sum(union, dim=(1, 2, 3))
    recall = ratio(I, G)
    precision = ratio(I, E)
    iou = ratio(I, U)

    return recall, precision, iou