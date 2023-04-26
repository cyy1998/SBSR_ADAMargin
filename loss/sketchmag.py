import torch
from torch.nn.functional import cross_entropy
import math

class SketchMagLoss:

    def __init__(self, scale: float = 15.0, margin: float = 0.35,sem_margin: float = 0.35,h=0.333, t_alpha=0.99, reduction: str = "mean",c_sim: str= "./extract_features/label_matrix.mat") -> None:
        self.scale = torch.tensor(scale,requires_grad=True)
        self.margin = margin
        self.eps = 1e-3
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.reduction = reduction
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.h=h
        self.t_alpha=t_alpha
        self.t=torch.zeros(1).cuda()
        self.batch_mean=torch.ones(1).cuda()*20
        self.batch_std=torch.ones(1).cuda()*100
        self.c_sim= torch.tensor(torch.load(c_sim)).cuda()
        self.c_sim=sem_margin*(torch.max(self.c_sim)-self.c_sim)/(torch.max(self.c_sim)-torch.min(self.c_sim))



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
        cosine = cosine.clamp(-1 + self.eps, 1 - self.eps)
        safe_norms=feature_norms.clone().detach()
        with torch.no_grad():
            mean = safe_norms.mean()
            std = safe_norms.std()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std = std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)  # 66% between -1, 1
        margin_scaler = margin_scaler * self.h  # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)

        m_arc = torch.zeros(targets.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, targets.reshape(-1, 1), 1.0)
        g_angular = self.margin * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi - self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(targets.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, targets.reshape(-1, 1), 1.0)
        g_add = self.margin + (self.margin * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # m_sem
        m_sem= torch.ones(targets.size()[0], cosine.size()[1], device=cosine.device)
        m_sem.scatter_(1, targets.reshape(-1, 1), 0.0)
        m_sem = self.c_sim[targets,:]*m_sem
        cosine = cosine+m_sem
        scaled_cosine_m = cosine * self.scale

        return cross_entropy(input=scaled_cosine_m, target=targets, reduction=self.reduction,label_smoothing=0.1)