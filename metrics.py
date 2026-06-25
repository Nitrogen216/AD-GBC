import torch

try:
    from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision
except ImportError:
    import numpy as np
    from scipy.ndimage import binary_erosion, distance_transform_edt

    def jc(result, reference):
        result = np.asarray(result).astype(bool)
        reference = np.asarray(reference).astype(bool)
        union = np.logical_or(result, reference).sum()
        if union == 0:
            return 1.0
        return np.logical_and(result, reference).sum() / union

    def dc(result, reference):
        result = np.asarray(result).astype(bool)
        reference = np.asarray(reference).astype(bool)
        size_sum = result.sum() + reference.sum()
        if size_sum == 0:
            return 1.0
        return 2.0 * np.logical_and(result, reference).sum() / size_sum

    def _surface_distances(result, reference):
        result = np.asarray(result).astype(bool)
        reference = np.asarray(reference).astype(bool)
        if not result.any() or not reference.any():
            return np.array([np.inf])
        result_border = np.logical_xor(result, binary_erosion(result))
        reference_border = np.logical_xor(reference, binary_erosion(reference))
        dt = distance_transform_edt(~reference_border)
        return dt[result_border]

    def hd(result, reference):
        forward = _surface_distances(result, reference)
        backward = _surface_distances(reference, result)
        return float(max(forward.max(), backward.max()))

    def hd95(result, reference):
        forward = _surface_distances(result, reference)
        backward = _surface_distances(reference, result)
        return float(np.percentile(np.hstack([forward, backward]), 95))

    def recall(result, reference):
        result = np.asarray(result).astype(bool)
        reference = np.asarray(reference).astype(bool)
        positives = reference.sum()
        if positives == 0:
            return 1.0
        return np.logical_and(result, reference).sum() / positives

    def specificity(result, reference):
        result = np.asarray(result).astype(bool)
        reference = np.asarray(reference).astype(bool)
        negatives = (~reference).sum()
        if negatives == 0:
            return 1.0
        return np.logical_and(~result, ~reference).sum() / negatives

    def precision(result, reference):
        result = np.asarray(result).astype(bool)
        reference = np.asarray(reference).astype(bool)
        predicted = result.sum()
        if predicted == 0:
            return 1.0
        return np.logical_and(result, reference).sum() / predicted


def iou_score(output, target):
    ### --- 新增：解包元组 --- ###
    # 检查 output 是否为元组 (在训练时，模型会返回 (seg_output, loss_intermediates))
    # 评估指标只关心 seg_output
    if isinstance(output, tuple):
        output = output[0]
    ### --- 修改结束 --- ###
    
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)
    return iou, dice


def dice_coef(output, target):
    if isinstance(output, tuple):
        output = output[0]

    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def indicators(output, target):
    if isinstance(output, tuple):
        output = output[0]
        
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    iou_ = jc(output_, target_)
    dice_ = dc(output_, target_)
    hd_ = hd(output_, target_)
    hd95_ = hd95(output_, target_)
    recall_ = recall(output_, target_)
    specificity_ = specificity(output_, target_)
    precision_ = precision(output_, target_)

    return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_
