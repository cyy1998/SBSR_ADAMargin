# Author: Changmao Cheng <chengchangmao@megvii.com>

import torch
from torch.nn.functional import cross_entropy


class TransferLoss:

    def __init__(self, alpha: float = 2.0, beta: float = 1.0) -> None:
        self.alpha=alpha
        self.beta=beta

    def __call__(self, mu: torch.Tensor, labels: torch.LongTensor, mean: torch.LongTensor) -> torch.Tensor:
        """call function as forward

        Args:
            logits (torch.Tensor): The predicted logits before softmax with shape of :math:`(N, C)`
            targets (torch.LongTensor): The ground-truth label long vector with shape of :math:`(N,)`

        Returns:
            torch.Tensor: loss
                the computed loss
        """
        means = mean[labels]
#         loss_va=None
#         for idx,label in enumerate(labels):
#             other= torch.full((mean.shape[0],),True,dtype=bool)
#             other[label]=False
#             other_means = mean[other]
#             loss_a=mu[idx]*other_means
#             loss_a=torch.sum(loss_a,1)
#             loss_a=torch.mean(loss_a)
#             if loss_va:
#                 loss_va=loss_va+loss_a
#             else:
#                 loss_va=loss_a
                
#         loss_va=loss_va/len(labels)
        loss_v1= self.alpha*((mu - means) * (mu - means))
        loss_v2 = torch.sum(loss_v1, 1)
        loss = 10*torch.mean(loss_v2)
        return loss