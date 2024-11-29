from torch import einsum, logsumexp, no_grad

def loss_fn(ref, pos, neg):
    if 'recon_loss' in ref:
        return contrast_recon_loss(ref, pos, neg)
    else:
        return info_nce(ref['z'], pos['z'], neg['z'], ref['temp'])

def contrast_recon_loss(ref, pos, neg):
    ref_z, ref_recon_loss, ref_temp = ref['z'], ref['recon_loss'], ref['temp']
    pos_z, pos_recon_loss, pos_temp = pos['z'], pos['recon_loss'], pos['temp']
    neg_z, neg_recon_loss, neg_temp = neg['z'], neg['recon_loss'], neg['temp']

    info_nce_loss = info_nce(ref_z, pos_z, neg_z, ref_temp)
    loss = ref_recon_loss + info_nce_loss
    return loss

def info_nce(ref, pos, neg, tau=1.0):
    # Calculate distances
    pos_dist = einsum("nd,nd->n", ref, pos) / tau
    neg_dist = einsum("nd,md->nm", ref, neg) / tau
    
    with no_grad():
        c, _ = neg_dist.max(dim=1, keepdim=True)
    c = c.detach()
    pos_dist = pos_dist - c.squeeze(1)
    neg_dist = neg_dist - c

    # Compute the losses
    pos_loss = -pos_dist.mean()
    neg_loss = logsumexp(neg_dist, dim=1).mean()
    print("pos_loss: ", pos_loss, "neg_loss: ", neg_loss)
    # Total loss is the sum of positive and negative losses
    return pos_loss + neg_loss
