# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.yolo.utils.metrics import OKS_SIGMA
from ultralytics.yolo.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

from .metrics import bbox_iou
from .tal import bbox2dist
import numpy as np

from . import dmtloss
import math



import torch.distributed as dist  


class TGLoss(nn.Module):  
    def __init__(self, feature_dim, k):  
        super(TGLoss, self).__init__()  
        self.TSDM = TargetSampleDiscriminabilityModule(feature_dim, k)  
        self.GDDM = GlobalDomainDifferenceModule(feature_dim)  

    


    def forward(self, source_features, target_features):  
        discriminability = self.TSDM(target_features)  
        GFDD = self.GDDM(source_features, target_features)  


        return self.compute_date_loss(discriminability, GFDD)  

    def compute_date_loss(self, discriminability, GFDD):  
        disc_loss = torch.mean(discriminability) 
        loss_GDD = torch.mean(GFDD)  

        


        return  loss_GDD, disc_loss


class TargetSampleDiscriminabilityModule(nn.Module):  
    def __init__(self, feature_dim, k):  
        super(TargetSampleDiscriminabilityModule, self).__init__()  
        self.k = k  
        self.fc = nn.Linear(2, 1)  # 输入维度为2  

    def forward(self, features):  
        # 同步特征数量  
        num_features = features.size(0)  
        num_features_tensor = torch.tensor([num_features], device=features.device)  
        # dist.all_reduce(num_features_tensor, op=dist.ReduceOp.MIN)  
        min_num_features = num_features_tensor.item()  

        # 调整 k 值  
        k = min(self.k, min_num_features - 1)  # 确保至少有一个其他样本  

        if k < 1:  
            # 处理极端情况  
            return torch.zeros(num_features, device=features.device)  

        # 计算距离  
        distances = torch.cdist(features, features)  

        # 计算 top-k 距离  
        topk_distances, _ = torch.topk(distances, k=k+1, largest=False)  # +1 because the first one is always 0 (self-distance)  
        inc_compactness = topk_distances[:, 1:].mean(dim=1)  

        # 计算平均距离  
        inc_separability = distances.sum(dim=1) / (num_features - 1)  # 除以 n-1 因为不包括自身  

        # 将密度和不确定性拼接  
        combined = torch.stack([inc_compactness, inc_separability], dim=1)  

        # 通过全连接层  
        discriminability = self.fc(combined)  
        return torch.abs(discriminability).squeeze()  

    # @staticmethod  
    # def sync_tensor(tensor):  
    #     rt = tensor.clone()  
    #     dist.all_reduce(rt, op=dist.ReduceOp.MIN)  
    #     return rt


class GlobalDomainDifferenceModule(nn.Module):   
    def __init__(self, feature_dim):  
        super(GlobalDomainDifferenceModule, self).__init__()  
        self.fc = nn.Linear(feature_dim, 1)  

    def forward(self, source_features, target_features):  
        # 获取每个域的特征均值  
        source_mean = source_features.mean(dim=0)  
        target_mean = target_features.mean(dim=0)  

        # 计算均值之间的距离(直接作为向量)  
        feature_diff = source_mean - target_mean  

        # 使用全连接层进一步处理特征  
        transferability = self.fc(feature_diff.view(1, -1))  

        return torch.abs(transferability).squeeze()
class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(
            pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True
        )
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = (
                self._df_loss(
                    pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape)
            * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)
            * wr
        ).mean(-1, keepdim=True)


class KeypointLoss(nn.Module):

    def __init__(self, sigmas) -> None:
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (
            pred_kpts[..., 1] - gt_kpts[..., 1]
        ) ** 2
        kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0)) / (
            torch.sum(kpt_mask != 0) + 1e-9
        )
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / (2 * self.sigmas) ** 2 / (area + 1e-9) / 2  # from cocoeval
        return kpt_loss_factor * ((1 - torch.exp(-e)) * kpt_mask).mean()


# Criterion class for computing Detection training losses
class v8DetectionLoss:

    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride[:3]  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(
            topk=10, num_classes=self.nc, alpha=0.5, beta=6.0
        )
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

        self.TG_loss = None  # 我们将在第一次调用时初始化 DATE 损失  









    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self.proj.type(pred_dist.dtype))
            )
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)


    def __call__(self, preds, batch):

        im_files = batch["im_file"]

        n_batch = next(
            (index for index, item in enumerate(im_files) if "Unlabeled" in item), None
        )

        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        daerror = []

        if len(preds) > 3 and n_batch != None:
            loss = torch.zeros(4, device=self.device)  # box, cls, dfl

            batchx = {
                "im_file": batch["im_file"][:n_batch],
                "ori_shape": batch["ori_shape"][:n_batch],
                "resized_shape": batch["resized_shape"][:n_batch],
                "img": batch["img"][:n_batch],
                "cls": batch["cls"],
                "bboxes": batch["bboxes"],
                "batch_idx": batch["batch_idx"],
            }

            batch = batchx
            daerror = preds[3]
            predsx = [
                preds[0][: int(n_batch)],
                preds[1][: int(n_batch)],
                preds[2][: int(n_batch)],
            ]
            preds = predsx

        preds = preds[:3]



        feats = preds[1] if isinstance(preds, tuple) else preds

        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )  # image size (h,w)

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            1,
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = (
            self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        )  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        # DA loss

        if len(daerror) > 0:
            tn = daerror.size()[0]

            vs = daerror[:n_batch]
            vt = daerror[n_batch:]

            feature_dim = daerror.size()[1]  # 特征维度  

            k = min(5, n_batch - 1)  # k 不能大于源域样本数减 1

            if self.TG_loss is None:  
                self.TG_loss = TGLoss(feature_dim, k).to(self.device)  

            # 计算 DATE 损失  
            gda_loss, tsd_loss = self.TG_loss(vs, vt)



            # 使用矩阵运算计算成对距离
            vs_repeat = vs.unsqueeze(1).repeat(1, tn - n_batch, 1)
            vt_repeat = vt.unsqueeze(0).repeat(n_batch, 1, 1)
            pairwise_distances = torch.norm(vs_repeat - vt_repeat, p=2, dim=2)
            # 对于 vt 中的每个样本,找到其到 vs 中所有样本的最短距离
            min_distances, _ = torch.min(pairwise_distances, dim=0)

            lda_loss = torch.sum(min_distances) / (tn - n_batch)




            # 组合所有损失  
            loss[3] = lda_loss*self.hyp.LLDA + gda_loss*self.hyp.LGDA + tsd_loss*self.hyp.LTSD





            


        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain



        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)








# Criterion class for computing training losses
class v8SegmentationLoss(v8DetectionLoss):

    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.nm = model.model[-1].nm  # number of masks
        self.overlap = model.args.overlap_mask




# Criterion class for computing training losses
class v8PoseLoss(v8DetectionLoss):

    def __init__(self, model):  # model must be de-paralleled
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = (
            torch.from_numpy(OKS_SIGMA).to(self.device)
            if is_pose
            else torch.ones(nkpt, device=self.device) / nkpt
        )
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)




class v8ClassificationLoss:

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = (
            torch.nn.functional.cross_entropy(preds, batch["cls"], reduction="sum") / 64
        )
        loss_items = loss.detach()
        return loss, loss_items
