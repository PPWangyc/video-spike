from torch import einsum, logsumexp, no_grad

def loss_fn_(ref, pos, neg, fix_temp=True):
    # MAE Loss
    if pos is None and neg is None:
        return {
            'loss': ref['recon_loss']
        }
    if 'recon_loss' in ref and 'temp' in ref:
        return contrast_recon_loss(ref, pos, neg,fix_temp)
    elif 'z' in ref and 'temp' in ref:
        if fix_temp:
            return info_nce(ref['z'], pos['z'], neg['z'])
        else:
            return info_nce(ref['z'], pos['z'], neg['z'], ref['temp'])
    elif 'z' in ref and 'recon_loss' in ref:
        return {
            'loss': (ref['recon_loss'] + pos['recon_loss'] + neg['recon_loss'])/3
        }
    else:
        raise ValueError('Invalid Loss input')
def contrast_recon_loss(ref, pos, neg,fix_temp=True):
    ref_z, ref_recon_loss, ref_temp = ref['z'], ref['recon_loss'], ref['temp']
    pos_z, pos_recon_loss, _ = pos['z'], pos['recon_loss'], pos['temp']
    neg_z, neg_recon_loss, _ = neg['z'], neg['recon_loss'], neg['temp']
    _temp = 1.0 if fix_temp else ref_temp
    info_nce_loss = info_nce(ref_z, pos_z, neg_z, _temp)
    mean_recon_loss = (ref_recon_loss + pos_recon_loss + neg_recon_loss) / 3
    loss = mean_recon_loss + info_nce_loss['loss']
    return {
        'loss': loss,
        'recon_loss': mean_recon_loss,
        'pos_loss': info_nce_loss['pos_loss'],
        'neg_loss': info_nce_loss['neg_loss'],
        'contrast_loss': info_nce_loss['loss']
    }

#
# CEBRA: Consistent EmBeddings of high-dimensional Recordings using Auxiliary variables
# © Mackenzie W. Mathis & Steffen Schneider (v0.4.0+)
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/blob/main/LICENSE.md
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Criterions for contrastive learning

Different criterions can be used for learning embeddings with CEBRA. The common
interface of criterions implementing the generalized InfoNCE metric is given by
:py:class:`BaseInfoNCE`.

Criterions are available for fixed and learnable temperatures, as well as different
similarity measures.

Note that criterions can have trainable parameters, which are automatically handled
by the training loops implemented in :py:class:`cebra.solver.base.Solver` classes.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@torch.jit.script
def dot_similarity(ref: torch.Tensor, pos: torch.Tensor,
                   neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cosine similarity the ref, pos and negative pairs

    Args:
        ref: The reference samples of shape `(n, d)`.
        pos: The positive samples of shape `(n, d)`.
        neg: The negative samples of shape `(n, d)`.

    Returns:
        The similarity between reference samples and positive samples of shape `(n,)`, and
        the similarities between reference samples and negative samples of shape `(n, n)`.
    """
    pos_dist = torch.einsum("ni,ni->n", ref, pos)
    neg_dist = torch.einsum("ni,mi->nm", ref, neg)
    return pos_dist, neg_dist


@torch.jit.script
def euclidean_similarity(
        ref: torch.Tensor, pos: torch.Tensor,
        neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Negative L2 distance between the ref, pos and negative pairs

    Args:
        ref: The reference samples of shape `(n, d)`.
        pos: The positive samples of shape `(n, d)`.
        neg: The negative samples of shape `(n, d)`.

    Returns:
        The similarity between reference samples and positive samples of shape `(n,)`, and
        the similarities between reference samples and negative samples of shape `(n, n)`.
    """
    ref_sq = torch.einsum("ni->n", ref**2)
    pos_sq = torch.einsum("ni->n", pos**2)
    neg_sq = torch.einsum("ni->n", neg**2)

    pos_cosine, neg_cosine = dot_similarity(ref, pos, neg)
    pos_dist = -(ref_sq + pos_sq - 2 * pos_cosine)
    neg_dist = -(ref_sq[:, None] + neg_sq[None] - 2 * neg_cosine)

    return pos_dist, neg_dist


@torch.jit.script
def infonce(
        pos_dist: torch.Tensor, neg_dist: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """InfoNCE implementation

    See :py:class:`BaseInfoNCE` for reference.

    Note:
        - The behavior of this function changed beginning in CEBRA 0.3.0.
        The InfoNCE implementation is numerically stabilized.
    """
    # with torch.no_grad():
    #     c, _ = neg_dist.max(dim=1, keepdim=True)
    # c = c.detach()
    # print(c.shape)
    # pos_dist = pos_dist - c.squeeze(1)
    # neg_dist = neg_dist - c
    align = (-pos_dist).mean()
    # print(align)
    uniform = torch.logsumexp(neg_dist, dim=1).mean()
    return align + uniform, align, uniform
    c_mean = c.mean()
    align_corrected = align - c_mean
    uniform_corrected = uniform + c_mean

    return align + uniform, align_corrected, uniform_corrected


class ContrastiveLoss(nn.Module):
    """Base class for contrastive losses.

    Note:
        - Added in 0.0.2.
    """

    def forward(
            self, ref: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the contrastive loss.

        Args:
            ref: The reference samples of shape `(n, d)`.
            pos: The positive samples of shape `(n, d)`.
            neg: The negative samples of shape `(n, d)`.
        """
        raise NotImplementedError()


class BaseInfoNCE(ContrastiveLoss):
    r"""Base class for all InfoNCE losses.

    Given a similarity measure :math:`\phi` which will be implemented by the subclasses
    of this class, the generalized InfoNCE loss is computed as

    .. math::

        \sum_{i=1}^n - \phi(x_i, y^{+}_i) + \log \sum_{j=1}^{n} e^{\phi(x_i, y^{-}_{ij})}

    where :math:`n` is the batch size, :math:`x` are the reference samples (``ref``),
    :math:`y^{+}` are the positive samples (``pos``) and :math:`y^{-}` are the negative
    samples (``neg``).

    """

    def _distance(self, ref: torch.Tensor, pos: torch.Tensor,
                  neg: torch.Tensor) -> Tuple[torch.Tensor]:
        """The similarity measure.

        Args:
            ref: The reference samples of shape `(n, d)`.
            pos: The positive samples of shape `(n, d)`.
            neg: The negative samples of shape `(n, d)`.

        Returns:
            The distance between reference samples and positive samples of shape `(n,)`, and
            the distances between reference samples and negative samples of shape `(n, n)`.

        """
        raise NotImplementedError()

    def forward(self, ref, pos,
                neg) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the InfoNCE loss.

        Args:
            ref: The reference samples of shape `(n, d)`.
            pos: The positive samples of shape `(n, d)`.
            neg: The negative samples of shape `(n, d)`.

        See Also:
            :py:class:`BaseInfoNCE`.
        """
        pos_dist, neg_dist = self._distance(ref, pos, neg)
        return infonce(pos_dist, neg_dist)


class FixedInfoNCE(BaseInfoNCE):
    """InfoNCE base loss with a fixed temperature.

    Attributes:
        temperature:
            The softmax temperature
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature


class LearnableInfoNCE(BaseInfoNCE):
    """InfoNCE base loss with a learnable temperature.

    Attributes:
        temperature:
            The current value of the learnable temperature parameter.
        min_temperature:
            The minimum temperature to use. Increase the minimum temperature
            if you encounter numerical issues during optimization.
    """

    def __init__(self,
                 temperature: float = 1.0,
                 min_temperature: Optional[float] = None):
        super().__init__()
        if min_temperature is None:
            self.max_inverse_temperature = math.inf
        else:
            self.max_inverse_temperature = 1.0 / min_temperature
        log_inverse_temperature = torch.tensor(
            math.log(1.0 / float(temperature)))
        self.log_inverse_temperature = nn.Parameter(log_inverse_temperature)
        self.min_temperature = min_temperature

    @torch.jit.export
    def _prepare_inverse_temperature(self) -> torch.Tensor:
        """Compute the current inverse temperature."""
        inverse_temperature = torch.exp(self.log_inverse_temperature)
        inverse_temperature = torch.clamp(inverse_temperature,
                                          max=self.max_inverse_temperature)
        return inverse_temperature

    @property
    def temperature(self) -> float:
        with torch.no_grad():
            return 1.0 / self._prepare_inverse_temperature().item()


class FixedCosineInfoNCE(FixedInfoNCE):
    r"""Cosine similarity function with fixed temperature.

    The similarity metric is given as

    .. math ::

        \phi(x, y) =  x^\top y  / \tau

    with fixed temperature :math:`\tau > 0`.

    Note that this loss function should typically only be used with normalized.
    This class itself does *not* perform any checks. Ensure that :math:`x` and
    :math:`y` are normalized.
    """

    @torch.jit.export
    def _distance(self, ref: torch.Tensor, pos: torch.Tensor,
                  neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_dist, neg_dist = dot_similarity(ref, pos, neg)
        return pos_dist / self.temperature, neg_dist / self.temperature


class FixedEuclideanInfoNCE(FixedInfoNCE):
    r"""L2 similarity function with fixed temperature.

    The similarity metric is given as

    .. math ::

        \phi(x, y) =  - \| x - y \| / \tau

    with fixed temperature :math:`\tau > 0`.
    """

    @torch.jit.export
    def _distance(self, ref: torch.Tensor, pos: torch.Tensor,
                  neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_dist, neg_dist = euclidean_similarity(ref, pos, neg)
        return pos_dist / self.temperature, neg_dist / self.temperature


class LearnableCosineInfoNCE(LearnableInfoNCE):
    r"""Cosine similarity function with a learnable temperature.

    Like :py:class:`FixedCosineInfoNCE`, but with a learnable temperature
    parameter :math:`\tau`.
    """

    @torch.jit.export
    def _distance(self, ref: torch.Tensor, pos: torch.Tensor,
                  neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inverse_temperature = self._prepare_inverse_temperature()
        pos, neg = dot_similarity(ref, pos, neg)
        return pos * inverse_temperature, neg * inverse_temperature


class LearnableEuclideanInfoNCE(LearnableInfoNCE):
    r"""L2 similarity function with fixed temperature.

    Like :py:class:`FixedEuclideanInfoNCE`, but with a learnable temperature
    parameter :math:`\tau`.
    """

    @torch.jit.export
    def _distance(self, ref: torch.Tensor, pos: torch.Tensor,
                  neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inverse_temperature = self._prepare_inverse_temperature()
        pos, neg = euclidean_similarity(ref, pos, neg)
        return pos * inverse_temperature, neg * inverse_temperature


# NOTE(stes): old aliases used in various locations in the codebase. Should be
# deprecated at some point.
InfoNCE = FixedCosineInfoNCE
InfoMSE = FixedEuclideanInfoNCE


class NCE(ContrastiveLoss):
    """Noise contrastive estimation (Gutman & Hyvarinen, 2012)

    Attributes:
        temperature (float): The softmax temperature
        negative_weight (float): Relative weight of the negative samples
        reduce (str): How to reduce the negative samples. Can be
            ``sum`` or ``mean``.
    """

    def __init__(self, temperature=1.0, negative_weight=1.0, reduce="mean"):
        super().__init__()
        self.temperature = temperature
        self.negative_weight = negative_weight
        assert reduce in ["mean", "sum"]
        self._reduce = getattr(torch, reduce)

    def forward(self, ref, pos, neg):
        """Compute the NCE loss.

        Args:
            ref: The reference samples of shape `(n, d)`.
            pos: The positive samples of shape `(n, d)`.
            neg: The negative samples of shape `(n, d)`.

        See Also:
            :py:class:`NCE`.
        """

        pos_dist = torch.einsum("ni,ni->n", ref, pos) / self.temperature
        neg_dist = torch.einsum("ni,mi->nm", ref, neg) / self.temperature

        align = F.logsigmoid(pos_dist)
        uniform = self._reduce(F.logsigmoid(-neg_dist), dim=1)

        return align + self.negative_weight * uniform, align, uniform

class loss_fn(nn.Module):
    def __init__(self, 
                 nce_type='learnable_cosine',
                 temperature=1.0):
        super().__init__()
        if nce_type == 'learnable_cosine':
            self.info_nce = LearnableCosineInfoNCE(temperature,None)
        elif nce_type == 'fixed_cosine':
            self.info_nce = FixedCosineInfoNCE(temperature)

    def forward(self, ref, pos, neg):
        if 'recon_loss' in ref:
            return contrast_recon_loss(ref, pos, neg)
        else:
            loss, pos_loss, neg_loss = self.info_nce(ref['z'], pos['z'], neg['z'])
            return {
                'loss': loss,
                'pos_loss': pos_loss,
                'neg_loss': neg_loss
            }
        
def info_nce(ref, pos, neg, tau=1.0):
    # Calculate distances
    pos_dist = einsum("nd,nd->n", ref, pos) / tau
    neg_dist = einsum("nd,md->nm", ref, neg) / tau
    
    # stabilize the computation
    with no_grad():
        c, _ = neg_dist.max(dim=1, keepdim=True)
        # c = torch.max(pos_dist, neg_dist.max(dim=1)[0]).unsqueeze(1)
    c = c.detach()
    pos_dist = pos_dist - c.squeeze(1)
    neg_dist = neg_dist - c

    # Compute the losses
    pos_loss = -pos_dist.mean()
    neg_loss = logsumexp(neg_dist, dim=1).mean()
    loss = pos_loss + neg_loss
    # Total loss is the sum of positive and negative losses
    return {
        'loss': loss,
        'pos_loss': pos_loss,
        'neg_loss': neg_loss
    }
