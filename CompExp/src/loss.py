from torch.nn.functional import cross_entropy, nll_loss


def mask_nll_loss(inp, target, mask):
    inp = inp.view(-1, inp.size(-1))
    target = target.view(-1)

    mask = mask.view(-1)

    loss = nll_loss(inp, target, reduction='none').masked_select(mask).mean()

    return loss


def mask_ce_loss(inp, target, mask):
    '''
    Calculate the cross entropy loss and a binary mask tensor describing the padding of the target tensor.

    Inputs:
        inp: tensor of raw values; size=(seq, batch, classes)
        target: tensor of classes; size=(seq, batch)
        mask: binary mask; size=(seq, batch)

    Outputs:
        loss: scalar
    '''

    # format the sizes
    inp = inp.view(-1, inp.size(-1))
    target = target.view(-1)

    mask = mask.view(-1)

    loss = cross_entropy(
        inp, target, reduction='none').masked_select(mask).mean()

    return loss
