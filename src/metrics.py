import torch

def psa(target, estimate):
    return torch.mean(torch.abs(estimate-target))

def sisdr(target, estimate):
    target *= torch.sum(target*estimate, dim=-1, keepdim=True) / torch.sum(target**2, dim=-1, keepdim=True)
    numerator = torch.sum(target**2, dim=-1)
    denominator = torch.sum((estimate-target)**2, dim=-1)
    retval = 10*torch.log10(numerator/denominator)
    return retval