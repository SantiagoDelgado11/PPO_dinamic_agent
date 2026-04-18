import torch

def has_svd(forward_op):
    return hasattr(forward_op, 'Ut') and hasattr(forward_op, 'Vt')


def has_pseudo_inverse(forward_op):
    return hasattr(forward_op, 'pseudo_inverse')
