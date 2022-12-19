
import torch
from torch.nn.functional import cross_entropy
import math


class ArcFaceLoss:

    def __init__(self, scale: float = 15.0, margin: float = 0.35,easy_margin=False, reduction: str = "mean") -> None:
        self.scale = torch.tensor(scale,requires_grad=True)
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.reduction = reduction
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.easy_margin=easy_margin

    def __call__(self, logits: torch.Tensor,feature_norms: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
        """call function as forward

        Args:
            logits (torch.Tensor): The predicted logits before softmax with shape of :math:`(N, C)`
            targets (torch.LongTensor): The ground-truth label long vector with shape of :math:`(N,)`

        Returns:
            torch.Tensor: loss
                the computed loss
        """

        #print(self.scale)
        cosine=logits
        sine=torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot_labels = torch.zeros_like(logits).scatter_(dim=1, index=targets.unsqueeze(1), value=1.0)
        output = (one_hot_labels * phi) + ((1.0 - one_hot_labels) * cosine)
        output *= self.scale
        return cross_entropy(input=output, target=targets, reduction=self.reduction,label_smoothing=0.1)