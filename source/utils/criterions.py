#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/utils/criterions.py
"""

import torch
import torch.nn.functional as F
from torch import distributions
from torch.nn.modules.loss import _Loss


class NormalKLLoss(_Loss):
    """
    NormalKLLoss
    """

    def __init__(self, reduction='mean'):
        super(NormalKLLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction

    def forward(self, q_mu, q_logvar, p_mu=None, p_logvar=None):
        """
        q_mu: (batch_size, latent_size)
        q_logvar: (batch_size, latent_size)
        """
        if p_mu is None:
            p_mu = torch.zeros_like(q_mu)
        if p_logvar is None:
            p_logvar = torch.zeros_like(q_logvar)

        q_norm = distributions.Normal(q_mu, q_logvar.exp().sqrt())
        p_norm = distributions.Normal(p_mu, p_logvar.exp().sqrt())
        kl = distributions.kl_divergence(q_norm, p_norm).sum(dim=1)

        if self.reduction == 'mean':
            kl = kl.mean()
        elif self.reduction == 'sum':
            kl = kl.sum()
        return kl


class CatKLLoss(_Loss):
    """
    CatKLLoss
    """

    def __init__(self, reduction='none'):
        super(CatKLLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction

    def forward(self, log_qy, log_py):
        """
        KL(qy|py) = Eq[qy * log(q(y) / p(y))]

        log_qy: (batch_size, latent_size)
        log_py: (batch_size, latent_size)
        """
        qy = torch.exp(log_qy)
        kl = torch.sum(qy * (log_qy - log_py), dim=1)

        if self.reduction == 'mean':
            kl = kl.mean()
        elif self.reduction == 'sum':
            kl = kl.sum()
        return kl


class NLLLoss(_Loss):
    """
    NLLLoss
    """

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super(NLLLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target, reduction=True):
        """
        input: (batch_size, max_len, vocab_size)
        target: (batch_size, max_len)
        """
        batch_size = input.size(0)
        nll = F.nll_loss(input=input.view(-1, input.size(-1)),
                        target=target.contiguous().view(-1),
                        weight=self.weight,
                        reduction='none')
        nll = nll.view(batch_size, -1).sum(dim=1)

        if reduction:
            if self.reduction == 'mean':
                nll = nll.mean()
            elif self.reduction == 'sum':
                nll = nll.sum()

        return nll


class KLDivLoss(_Loss):
    """
    KLDivLoss
    """

    def __init__(self, reduction='mean'):
        super(KLDivLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean', 'batchmean']
        self.reduction = reduction

    def forward(self, p, q):
        """
        p: (batch_size, max_len)
        q: (batch_size, max_len)
        """
        return F.kl_div(torch.log(p + 1e-10), q, reduction=self.reduction)


class JSDivLoss(_Loss):
    """
    JSDivLoss
    """

    def __init__(self, reduction='mean'):
        super(JSDivLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean', 'batchmean']
        self.reduction = reduction
        self.kl_loss = KLDivLoss(reduction=self.reduction)

    def forward(self, p, q):
        """
        p: (batch_size, max_len)
        q: (batch_size, max_len)
        """
        mid = (p + q)/2
        loss = 0 
        loss += self.kl_loss(mid, p)
        loss += self.kl_loss(mid, q)
        loss /= 2
        return loss

class SymmetryKLDivLoss(_Loss):
    """
    SymmetryKLDivLoss
    """

    def __init__(self, reduction='mean'):
        super(SymmetryKLDivLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean', 'batchmean']
        self.reduction = reduction
        self.kl_loss = KLDivLoss(reduction=self.reduction)

    def forward(self, p, q):
        """
        p: (batch_size, max_len)
        q: (batch_size, max_len)
        """
        loss = 0 
        loss += self.kl_loss(p, q)
        loss += self.kl_loss(q, p)
        loss /= 2
        return loss

class MaskBCELoss(_Loss):
    """
    MaskBCELoss
    """

    def __init__(self, reduction='mean'):
        super(MaskBCELoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction

    def forward(self, input, target, mask=None):
        """
        input: (batch_size, max_len)
        target: (batch_size, max_len)
        mask: (batch_size, max_len)
        """
        bce = F.binary_cross_entropy(input=input,
                                     target=target,
                                     reduction='none')
        if mask is not None:
            bce *= mask.float()

        bce = bce.sum(dim=1)

        if self.reduction == 'mean':
            bce = bce.mean()
        elif self.reduction == 'sum':
            bce = bce.sum()
        return bce


class RedundancyLoss(_Loss):
    """
    RedundancyLoss
    """

    def __init__(self):
        super(RedundancyLoss, self).__init__()

    def forward(self, A):
        """
        forward
        """
        I = torch.eye(A.size(1))
        if A.is_cuda:
            I = I.cuda()
        norm = torch.bmm(A, A.transpose(1, 2)) - I
        norm = torch.sum(
            torch.sum(norm.pow(2), dim=2), dim=1)  # ** 0.5
        loss = norm.mean()
        return loss


class BOWLoss(_Loss):
    def __init__(self, ignore_index=-1, reduction='mean'):
        super(BOWLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, target):
        """
        forward
        """
        label = torch.zeros_like(logits).scatter_add(
            1, target, 1.0*torch.ones_like(target, dtype=torch.float))
        if self.ignore_index >= 0:
            label[:, self.ignore_index] = 0
        loss = -label*logits
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.sum() / label.sum()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'batchmean':
            return loss.sum(1).mean()
        else:
            raise ValueError("No reduction like %s" % (self.reduction))


class ContrastiveBOWLoss(_Loss):
    def __init__(self, ignore_index=-1):
        super(ContrastiveBOWLoss, self).__init__()
        self.ignore_index = ignore_index
        self.bow_loss = BOWLoss(ignore_index=ignore_index, reduction='mean')

    def forward(self, logits, pos_target, neg_target):
        """
        forward
        """
        logits = logits.scatter(-1, pos_target.new_zeros((logits.size(0),1), dtype=torch.int64), value=-1e10)
        pos_logits = torch.gather(logits, -1, pos_target)
        neg_logits = torch.gather(logits, -1, neg_target)
        pos_len = pos_logits.size(-1)
        logits = torch.cat([pos_logits, neg_logits], -1)
        logits = F.log_softmax(logits, -1)
        logits = logits[:, :pos_len]
        label =  pos_target.ne(0).to(torch.float)
        loss = -label*logits
        loss = loss.sum(-1) / label.sum(-1)
        loss = loss.mean()
        return loss
