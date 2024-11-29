from torch import einsum, logsumexp, no_grad

def info_nce(ref, pos, neg, tau=1.0):
    # Calculate distances
    pos_dist = einsum("nd,nd->n", ref, pos) / tau
    neg_dist = einsum("nd,md->nm", ref, neg) / tau
    
    # Subtract the maximum value for numerical stability before the softmax
    # with no_grad():
    c, _ = neg_dist.max(dim=1)
    pos_dist = pos_dist - c.detach()
    neg_dist = neg_dist - c.detach()

    # Compute the losses
    pos_loss = -pos_dist.mean()
    neg_loss = logsumexp(neg_dist, dim=1).mean()

    # Total loss is the sum of positive and negative losses
    return pos_loss + neg_loss
