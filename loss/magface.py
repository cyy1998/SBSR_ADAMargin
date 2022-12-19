# Author: Changmao Cheng <chengchangmao@megvii.com>

import torch
from torch.nn.functional import cross_entropy


class MagFaceLoss:
    r"""Additive Margin Softmax loss as described in the paper `"Additive Margin Softmax for Face Verification"
    <https://arxiv.org/abs/1801.05599>`_. AM Softmax Loss is identical to `CosFace <https://arxiv.org/abs/1801.09414>`_.

    .. math::

        \frac{1}{N}\sum_i-log(\frac{e^{s\cdot (\cos\theta_{y_i,i}-m)}}{e^{s\cdot (\cos\theta_{y_i,i}-m)}+
        \sum_{k\neq y_i}e^{s\cdot \cos\theta_{k,i}}})

    where :math:`s` is the scale factor, :math:`m` denotes margin and :math:`i` indexes the :math:`i`-th sample.

    .. note::
        We do not do embedding normalization, weight normalization or linear transform in this function. This function is
        a post-process. Please use :meth:`~megrec.torch_model.modules.TransformEmbeddingToLogit` function first to convert
        embeddings into logits.

    Args:
        scale (float): Specifies the scale factor. Default: 64.0
        margin (float): Margin value. Default: 0.35
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    """

    def __init__(self, scale: float = 15.0, l_a=10,u_a=110,l_m=0.45,u_m=0.8,lamada=20, reduction: str = "mean") -> None:
        self.scale = torch.tensor(scale,requires_grad=True)
        self.l_a=l_a
        self.u_a=u_a
        self.l_m=l_m
        self.u_m=u_m
        self.lamada=lamada
        self.reduction = reduction
        self.mm=0.3

    def calc_loss_G(self, x_norm):
        g = 1 / (self.u_a ** 2) * x_norm + 1 / (x_norm)
        return torch.mean(g)

    def __call__(self, logits: torch.Tensor,feature_norms: torch.Tensor ,targets: torch.LongTensor) -> torch.Tensor:
        """call function as forward

        Args:
            logits (torch.Tensor): The predicted logits before softmax with shape of :math:`(N, C)`
            targets (torch.LongTensor): The ground-truth label long vector with shape of :math:`(N,)`

        Returns:
            torch.Tensor: loss
                the computed loss
        """

        #print(self.scale)
        margin = (self.u_m - self.l_m) / (self.u_a - self.l_a) * (feature_norms - self.l_a) + self.l_m
        loss_g = self.calc_loss_G(feature_norms)
        one_hot_labels = torch.zeros_like(logits).scatter_(dim=1, index=targets.unsqueeze(1), value=1.0)
        resverse_one_hot_labels=1-one_hot_labels
        am_logits = self.scale * (logits - one_hot_labels * margin)
        #am_logits = self.scale * (logits)
        loss= cross_entropy(input=am_logits, target=targets, reduction=self.reduction,label_smoothing=0.1)
        return loss+self.lamada*loss_g