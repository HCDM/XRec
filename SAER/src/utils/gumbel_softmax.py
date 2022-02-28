import torch
import torch.nn.functional as F


def gumbel_softmax(log_probs, temperature=0.5, ST=False, eps=1e-20):
    '''
    Inputs:
        log_probs: log probabilities shape=(*, n_class)
        ST: straight-forward
    Outputs:
        gumbel_softmax: (*, n_class)
    '''
    shape = log_probs.size()

    U = torch.rand(shape, device=log_probs.device)
    gumbel_samples = -torch.log(-torch.log(U + eps) + eps)
    y = log_probs + gumbel_samples
    y = F.softmax(y / temperature, dim=-1)

    if ST:
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y

    return y
