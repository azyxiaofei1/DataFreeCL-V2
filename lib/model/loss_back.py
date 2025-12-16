import torch
from torch.nn import functional as F
import torch.nn as nn
from lib.utils import to_one_hot
from torch.nn.modules.loss import _Loss


class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, output, label, reduction='mean'):
        loss = F.cross_entropy(output, label, reduction=reduction)
        return loss


class CrossEntropy_binary(nn.Module):
    def __init__(self):
        super(CrossEntropy_binary, self).__init__()

    def forward(self, output, label, reduction='mean'):
        binary_targets = to_one_hot(label.cpu(), output.size(1)).to(label.device)
        loss_cls = F.binary_cross_entropy_with_logits(input=output, target=binary_targets,
                                                      reduction='none'
                                                      ).sum(dim=1)
        if "mean" in reduction:
            loss_cls = loss_cls.mean()
        return loss_cls


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def kronecker(A, B):
    AB = torch.einsum("ab,cd->acbd", A, B)
    AB = AB.view(A.size(0) * B.size(0), A.size(1) * B.size(1))
    return AB


def get_class_wise_cc_matrix(f1, f2, labels, allowed_labels, is_nograd=True):
    if is_nograd:
        f1, f2 = f1.detach(), f2.detach()
    feature_dim = f1.shape[1]
    class_wise_cc_matrix = None
    batch_class_sample_num_squ = None
    class_eye_mt = (-1 * torch.eye(feature_dim)).cuda()
    for label_num in allowed_labels:
        indices = torch.where(labels == label_num)[0]
        if len(indices) > 0:
            label_num_f1 = f1[indices]
            label_num_f2 = f2[indices]
            label_sample_size = label_num_f1.shape[0]
            label_num_f1_detach = label_num_f1.detach()
            label_num_f2_detach = label_num_f2.detach()

            kronecker_prod = kronecker(label_num_f1, label_num_f2)
            cc_matrix = kronecker_prod.reshape([label_sample_size, label_sample_size, feature_dim, feature_dim])
            cc_matrix = torch.sum(cc_matrix, dim=1)
            class_cc_matrix = torch.sum(cc_matrix, dim=0)

            f1_f1_prod = label_num_f1_detach.T @ label_num_f1_detach
            f2_f2_prod = label_num_f2_detach.T @ label_num_f2_detach
            f1_f1_sum_diag = torch.diagonal(f1_f1_prod) * label_sample_size
            f2_f2_sum_diag = torch.diagonal(f2_f2_prod) * label_sample_size
            f1_f1_sum_diag = torch.sqrt(f1_f1_sum_diag)
            f2_f2_sum_diag = torch.sqrt(f2_f2_sum_diag)
            f1_f2_matrix = torch.mm(f1_f1_sum_diag.unsqueeze(0).T, f2_f2_sum_diag.unsqueeze(0))

            class_cc_matrix = class_cc_matrix / f1_f2_matrix
            class_cc_matrix = class_cc_matrix + class_eye_mt
            sample_num = label_sample_size ** 2
            if class_wise_cc_matrix is None:
                class_wise_cc_matrix = class_cc_matrix.unsqueeze(0)
                batch_class_sample_num_squ = torch.tensor([sample_num])
            else:
                class_wise_cc_matrix = torch.cat([class_wise_cc_matrix, class_cc_matrix.unsqueeze(0)], dim=0)
                batch_class_sample_num_squ = torch.cat([batch_class_sample_num_squ, torch.tensor([sample_num])])
    return class_wise_cc_matrix, batch_class_sample_num_squ


def sup_barlowtwin_class_wise_loss(f1, f2, labels, contra_loss_allowed_labels, alpha=1.):
    feature_dim = f1.shape[1]
    class_wise_cc_matrix = None
    batch_class_sample_num_squ = None
    loss = torch.tensor(0.).cuda()
    class_eye_mt = (-1 * torch.eye(feature_dim)).cuda()
    for label_num in contra_loss_allowed_labels:
        indices = torch.where(labels == label_num)[0]
        if len(indices) > 0:
            label_num_f1 = f1[indices]
            label_num_f2 = f2[indices]
            label_sample_size = label_num_f1.shape[0]
            label_num_f1_detach = label_num_f1.detach()
            label_num_f2_detach = label_num_f2.detach()

            kronecker_prod = kronecker(label_num_f1, label_num_f2)
            cc_matrix = kronecker_prod.reshape([label_sample_size, label_sample_size, feature_dim, feature_dim])
            cc_matrix = torch.sum(cc_matrix, dim=1)
            class_cc_matrix = torch.sum(cc_matrix, dim=0)

            f1_f1_prod = label_num_f1_detach.T @ label_num_f1_detach
            f2_f2_prod = label_num_f2_detach.T @ label_num_f2_detach
            f1_f1_sum_diag = torch.diagonal(f1_f1_prod) * label_sample_size
            f2_f2_sum_diag = torch.diagonal(f2_f2_prod) * label_sample_size
            f1_f1_sum_diag = torch.sqrt(f1_f1_sum_diag)
            f2_f2_sum_diag = torch.sqrt(f2_f2_sum_diag)
            f1_f2_matrix = torch.mm(f1_f1_sum_diag.unsqueeze(0).T, f2_f2_sum_diag.unsqueeze(0))

            class_cc_matrix = class_cc_matrix / f1_f2_matrix
            class_cc_matrix = class_cc_matrix + class_eye_mt
            sample_num = label_sample_size
            if class_wise_cc_matrix is None:
                class_wise_cc_matrix = class_cc_matrix.unsqueeze(0)
                batch_class_sample_num_squ = torch.tensor([sample_num])
            else:
                class_wise_cc_matrix = torch.cat([class_wise_cc_matrix, class_cc_matrix.unsqueeze(0)], dim=0)
                batch_class_sample_num_squ = torch.cat([batch_class_sample_num_squ, torch.tensor([sample_num])])
            class_cc_matrix = class_cc_matrix.pow_(2)
            on_diag = torch.diagonal(class_cc_matrix).sum()
            off_diag = off_diagonal(class_cc_matrix).sum()
            loss_temp = (on_diag + alpha * off_diag) / sample_num
            loss += loss_temp
    return loss, class_wise_cc_matrix, batch_class_sample_num_squ


def sup_barlowtwins_kd_class_wise_loss(teacher_f1, teacher_f2, stu_f1, stu_f2, labels, kd_loss_allowed_labels,
                                       use_rebalance_kd=True, alpha=1.):
    teach_cc_matrix, batch_class_sample_num_squ = get_class_wise_cc_matrix(teacher_f1, teacher_f2, labels,
                                                                           allowed_labels=kd_loss_allowed_labels)
    stu_class_wise_cc_matrix, _ = get_class_wise_cc_matrix(stu_f1, stu_f2, labels,
                                                           allowed_labels=kd_loss_allowed_labels,
                                                           is_nograd=False)
    assert len(teach_cc_matrix.shape) == 3 and len(batch_class_sample_num_squ.shape) == 1
    assert teach_cc_matrix.size(0) == stu_class_wise_cc_matrix.size(0) and \
           teach_cc_matrix.size(0) == batch_class_sample_num_squ.size(0)
    assert teach_cc_matrix.size(1) == stu_class_wise_cc_matrix.size(1)
    delta_cc_matrix = teach_cc_matrix - stu_class_wise_cc_matrix
    delta_cc_matrix = delta_cc_matrix.pow_(2)
    batch_class_sample_num_squ = batch_class_sample_num_squ.contiguous().view(-1, 1)
    batch_class_sample_num_squ = batch_class_sample_num_squ.unsqueeze(1)
    delta_cc_matrix /= batch_class_sample_num_squ
    delta_cc_matrix = torch.sum(delta_cc_matrix, dim=0)
    on_diag = torch.diagonal(delta_cc_matrix).sum()
    off_diag = off_diagonal(delta_cc_matrix).sum()
    kd_loss = on_diag + alpha * off_diag
    return kd_loss


def get_cc_matrix(f1, f2, labels, allowed_labels, is_nograd=True):
    if is_nograd:
        f1, f2 = f1.detach(), f2.detach()
    feature_dim = f1.shape[1]
    class_wise_cc_matrix = None
    batch_class_sample_num_squ = None
    class_eye_mt = (-1 * torch.eye(feature_dim)).cuda()
    f1_f1_diag = None
    f2_f2_diag = None
    for label_num in allowed_labels:
        indices = torch.where(labels == label_num)[0]
        if len(indices) > 0:
            label_num_f1 = f1[indices]
            label_num_f2 = f2[indices]
            label_sample_size = label_num_f1.shape[0]
            label_num_f1_detach = label_num_f1.detach()
            label_num_f2_detach = label_num_f2.detach()

            kronecker_prod = kronecker(label_num_f1, label_num_f2)
            cc_matrix = kronecker_prod.reshape([label_sample_size, label_sample_size, feature_dim, feature_dim])
            cc_matrix = torch.sum(cc_matrix, dim=1)
            class_cc_matrix = torch.sum(cc_matrix, dim=0)

            f1_f1_prod = label_num_f1_detach.T @ label_num_f1_detach
            f2_f2_prod = label_num_f2_detach.T @ label_num_f2_detach
            f1_f1_label_sum_diag = torch.diagonal(f1_f1_prod) * label_sample_size
            f2_f2_label_sum_diag = torch.diagonal(f2_f2_prod) * label_sample_size

            sample_num = label_sample_size
            if class_wise_cc_matrix is None:
                class_wise_cc_matrix = class_cc_matrix
                batch_class_sample_num_squ = torch.tensor([sample_num])
                f1_f1_diag = f1_f1_label_sum_diag
                f2_f2_diag = f2_f2_label_sum_diag
            else:
                class_wise_cc_matrix += class_cc_matrix
                f1_f1_diag += f1_f1_label_sum_diag
                f2_f2_diag += f2_f2_label_sum_diag
                batch_class_sample_num_squ = torch.cat([batch_class_sample_num_squ, torch.tensor([sample_num])])
    f1_f1_diag = torch.sqrt(f1_f1_diag)
    f2_f2_diag = torch.sqrt(f2_f2_diag)
    f1_f2_matrix = torch.mm(f1_f1_diag.unsqueeze(0).T, f2_f2_diag.unsqueeze(0))
    class_wise_cc_matrix = class_wise_cc_matrix / f1_f2_matrix
    class_wise_cc_matrix = class_wise_cc_matrix + class_eye_mt
    return class_wise_cc_matrix, batch_class_sample_num_squ


def sup_barlowtwin_loss(f1, f2, labels, contra_loss_allowed_labels, alpha=1.):
    feature_dim = f1.shape[1]
    class_wise_cc_matrix = None
    batch_class_sample_num_squ = None
    class_eye_mt = (-1 * torch.eye(feature_dim)).cuda()
    f1_f1_diag = None
    f2_f2_diag = None
    for label_num in contra_loss_allowed_labels:
        indices = torch.where(labels == label_num)[0]
        if len(indices) > 0:
            label_num_f1 = f1[indices]
            label_num_f2 = f2[indices]
            label_sample_size = label_num_f1.shape[0]
            label_num_f1_detach = label_num_f1.detach()
            label_num_f2_detach = label_num_f2.detach()

            kronecker_prod = kronecker(label_num_f1, label_num_f2)
            cc_matrix = kronecker_prod.reshape([label_sample_size, label_sample_size, feature_dim, feature_dim])
            cc_matrix = torch.sum(cc_matrix, dim=1)
            class_cc_matrix = torch.sum(cc_matrix, dim=0)

            f1_f1_prod = label_num_f1_detach.T @ label_num_f1_detach
            f2_f2_prod = label_num_f2_detach.T @ label_num_f2_detach
            f1_f1_label_sum_diag = torch.diagonal(f1_f1_prod) * label_sample_size
            f2_f2_label_sum_diag = torch.diagonal(f2_f2_prod) * label_sample_size

            sample_num = label_sample_size
            if class_wise_cc_matrix is None:
                class_wise_cc_matrix = class_cc_matrix
                batch_class_sample_num_squ = torch.tensor([sample_num])
                f1_f1_diag = f1_f1_label_sum_diag
                f2_f2_diag = f2_f2_label_sum_diag
            else:
                class_wise_cc_matrix += class_cc_matrix
                f1_f1_diag += f1_f1_label_sum_diag
                f2_f2_diag += f2_f2_label_sum_diag
                batch_class_sample_num_squ = torch.cat([batch_class_sample_num_squ, torch.tensor([sample_num])])
    f1_f1_diag = torch.sqrt(f1_f1_diag)
    f2_f2_diag = torch.sqrt(f2_f2_diag)
    f1_f2_matrix = torch.mm(f1_f1_diag.unsqueeze(0).T, f2_f2_diag.unsqueeze(0))
    class_wise_cc_matrix = class_wise_cc_matrix / f1_f2_matrix
    class_wise_cc_matrix = class_wise_cc_matrix + class_eye_mt
    class_wise_cc_matrix = class_wise_cc_matrix.pow_(2)
    on_diag = torch.diagonal(class_wise_cc_matrix).sum()
    off_diag = off_diagonal(class_wise_cc_matrix).sum()
    loss = (on_diag + alpha * off_diag)
    return loss, class_wise_cc_matrix, batch_class_sample_num_squ


def sup_barlowtwins_kd_loss(teacher_f1, teacher_f2, stu_f1, stu_f2, labels, kd_loss_allowed_labels,
                            use_rebalance_kd=True, alpha=1.):
    teach_cc_matrix, _ = get_cc_matrix(teacher_f1, teacher_f2, labels,
                                                                allowed_labels=kd_loss_allowed_labels)
    stu_class_wise_cc_matrix, _ = get_cc_matrix(stu_f1, stu_f2, labels,
                                                allowed_labels=kd_loss_allowed_labels,
                                                is_nograd=False)
    delta_cc_matrix = teach_cc_matrix - stu_class_wise_cc_matrix
    delta_cc_matrix = delta_cc_matrix.pow_(2)
    on_diag = torch.diagonal(delta_cc_matrix).sum()
    off_diag = off_diagonal(delta_cc_matrix).sum()
    kd_loss = on_diag + alpha * off_diag
    return kd_loss


# 软目标交叉熵
class SoftTarget_CrossEntropy(nn.Module):
    def __init__(self, mean=True):
        super().__init__()
        self.mean = mean

    def forward(self, output, soft_target, kd_temp):
        assert len(output) == len(soft_target)
        log_prob = torch.nn.functional.log_softmax(output / kd_temp, dim=1)
        if self.mean:
            loss = -torch.sum(log_prob * soft_target) / len(soft_target)
        else:
            loss = -torch.sum(log_prob * soft_target)
        return loss


class loss_fn_kd_KL(nn.Module):
    def __init__(self):
        super(loss_fn_kd_KL, self).__init__()

    def forward(self, scores, target_scores, T=2., reduction='mean'):
        log_scores = F.log_softmax(scores / T, dim=1)
        targets = F.softmax(target_scores / T, dim=1)
        criterion = torch.nn.KLDivLoss(reduction="none")
        loss_cls = criterion(log_scores, targets).sum(dim=1)
        if 'mean' in reduction:
            loss_cls = loss_cls.mean()
        return loss_cls


def loss_fn_kd(scores, target_scores, T=2., reduction="mean"):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    device = scores.device

    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)

    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    # n = scores.size(1)
    assert len(scores) == len(target_scores) and scores.size(1) == target_scores.size(1)
    # if n > target_scores.size(1):
    #     n_batch = scores.size(0)
    #     zeros_to_add = torch.zeros(n_batch, n - target_scores.size(1))
    #     zeros_to_add = zeros_to_add.to(device)
    #     targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    KD_loss_unnorm = -(targets_norm * log_scores_norm)
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)  # --> sum over classes
    if reduction == "mean":
        KD_loss_unnorm = KD_loss_unnorm.mean()  # --> average over batch

    # normalize
    KD_loss = KD_loss_unnorm * T ** 2

    return KD_loss


def loss_fn_kd_binary(scores, target_scores, T=2., reduction="mean"):
    """Compute binary knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""

    device = scores.device

    scores_norm = torch.sigmoid(scores / T)
    targets_norm = torch.sigmoid(target_scores / T)
    assert len(scores) == len(target_scores) and scores.size(1) == target_scores.size(1)
    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    # n = scores.size(1)
    # if n > target_scores.size(1):
    #     n_batch = scores.size(0)
    #     zeros_to_add = torch.zeros(n_batch, n - target_scores.size(1))
    #     zeros_to_add = zeros_to_add.to(device)
    #     targets_norm = torch.cat([targets_norm, zeros_to_add], dim=1)

    # Calculate distillation loss
    KD_loss_unnorm = -(targets_norm * torch.log(scores_norm) + (1 - targets_norm) * torch.log(1 - scores_norm))
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)  # --> sum over classes
    if reduction == "mean":
        KD_loss_unnorm = KD_loss_unnorm.mean()  # --> average over batch

    # normalize
    KD_loss = KD_loss_unnorm * T ** 2

    return KD_loss


def compute_cls_distill_binary_loss(labels, output, classes_per_task,
                                    pre_model_output_for_distill):
    binary_targets = to_one_hot(labels.cpu(), output.size(1)).to(labels.device)
    if pre_model_output_for_distill is not None:
        binary_targets = binary_targets[:, -classes_per_task:]
        binary_targets = torch.cat([torch.sigmoid(pre_model_output_for_distill), binary_targets], dim=1)
    predL = None if labels is None else F.binary_cross_entropy_with_logits(
        input=output, target=binary_targets, reduction='none'
    ).sum(dim=1).mean()  # --> sum over classes, then average over batch
    return predL


def compute_distill_binary_loss(output_for_distill, pre_model_output_for_distill):
    binary_targets = torch.sigmoid(pre_model_output_for_distill)
    predL = F.binary_cross_entropy_with_logits(input=output_for_distill, target=binary_targets, reduction='none'
                                               ).sum(dim=1).mean()  # --> sum over classes, then average over batch
    return predL


def mixup_criterion_iCaRL(pred, y_a, y_b, lam, classes_per_task):
    return (lam * compute_cls_binary_loss(y_a, pred, classes_per_task) +
            (1 - lam) * compute_cls_binary_loss(y_b, pred, classes_per_task)).mean()


def ori_criterion_iCaRL(ori_imgs_output, y, classes_per_task):
    return compute_cls_binary_loss(y, ori_imgs_output, classes_per_task)
    pass


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return (lam * criterion(pred, y_a, reduction='none') +
            (1 - lam) * criterion(pred, y_b, reduction='none')).mean()


def compute_cls_binary_loss(labels, output, classes_per_task):
    binary_targets = to_one_hot(labels.cpu(), output.size(1)).to(labels.device)
    binary_targets = binary_targets[:, -classes_per_task:]
    output_for_newclass_cls = output[:, -classes_per_task:]
    predL = F.binary_cross_entropy_with_logits(
        input=output_for_newclass_cls, target=binary_targets, reduction='none'
    ).sum(dim=1)  # --> sum over classes, then average over batch
    return predL


def compute_distill_loss(output_for_distill, previous_task_model_output, temp=1., reduction='mean'):
    # distill_previous_task_active_classes_num: dpt_active_classes_num
    distill_loss = loss_fn_kd(output_for_distill, previous_task_model_output, temp,
                              reduction=reduction)
    '''if self.cfg.TRAIN.DISTILL.softmax_sigmoid == 0:
        distill_loss = loss_fn_kd(output_for_distill, previous_task_model_output, temp,
                                  reduction=reduction) * (temp ** 2)
    elif self.cfg.TRAIN.DISTILL.softmax_sigmoid == 1:
        distill_loss = loss_fn_kd_binary(output_for_distill, previous_task_model_output,
                                         temp,
                                         reduction=reduction) * (temp ** 2)
    else:
        loss_fn_kd_KL_forward = loss_fn_kd_KL()
        distill_loss = loss_fn_kd_KL_forward(output_for_distill, previous_task_model_output,
                                             T=temp, reduction=reduction) * (temp ** 2)'''
    return distill_loss


class BalancedSoftmax(_Loss):
    """
    Balanced Softmax Loss
    """

    def __init__(self, sample_per_class):
        super(BalancedSoftmax, self).__init__()
        self.sample_per_class = torch.tensor(sample_per_class)

    def forward(self, input, label, reduction='mean'):
        return balanced_softmax_loss(label, input, self.sample_per_class, reduction)


def balanced_softmax_loss(labels, logits, sample_per_class, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


class dive_loss(_Loss):
    def __init__(self, sample_per_class, alpha, t=3., p=0.5):
        super(dive_loss, self).__init__()
        self.sample_per_class = torch.tensor(sample_per_class)
        self.alpha = alpha
        self.t = t
        self.p = p

    def forward(self, input, label, teacher_model_output_for_distill, reduction='mean'):
        bsce_loss = balanced_softmax_loss(label, input, self.sample_per_class, reduction)
        teacher_model_output_for_distill_logit = F.softmax(teacher_model_output_for_distill / self.t, dim=1)
        teacher_model_output_for_distill_logit = torch.pow(teacher_model_output_for_distill_logit, self.p)
        teacher_model_output_for_distill_logit = F.softmax(teacher_model_output_for_distill_logit, dim=1)
        log_scores = F.log_softmax(input / self.t, dim=1)
        criterion = torch.nn.KLDivLoss(reduction="none")
        dive_kd_loss = criterion(log_scores, teacher_model_output_for_distill_logit).sum(dim=1)
        if 'mean' in reduction:
            dive_kd_loss = dive_kd_loss.mean()
        return (1 - self.alpha) * bsce_loss + self.alpha * dive_kd_loss
