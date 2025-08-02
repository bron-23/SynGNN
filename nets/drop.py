'''
    Add `extra_repr` into DropPath implemented by timm 
    for displaying more info.
'''


import torch
import torch.nn as nn
from e3nn import o3
import torch.nn.functional as F
from timm.models.layers import drop_path as timm_drop_path # <--- 添加这一行


class IdentityWithBatch(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch=None):
        return x


class GraphDropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(GraphDropPath, self).__init__()
        self.drop_prob = drop_prob if drop_prob is not None else 0.0

    def forward(self, x, batch):
        if self.drop_prob == 0.0 or not self.training:
            return x

        # Determine per-graph keep probability
        batch_size = int(batch.max().item() + 1)
        # Create a dummy tensor of shape (batch_size, 1) for timm's drop_path
        # to generate a (batch_size,) keep_mask. drop_path scales by 1/(1-p) if kept.
        dummy_input_for_mask = torch.ones(batch_size, 1, dtype=x.dtype, device=x.device)
        keep_mask_per_graph = timm_drop_path(dummy_input_for_mask, self.drop_prob, True)  # True for training behavior

        # Reshape keep_mask to broadcast: (batch_size, 1, ..., 1)
        keep_mask_per_graph_reshaped = keep_mask_per_graph.view(batch_size, *((1,) * (x.ndim - 1)))

        # Apply mask
        out = x * keep_mask_per_graph_reshaped[batch]
        return out

    def extra_repr(self):
        return 'drop_prob={}'.format(self.drop_prob)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self):
        return 'drop_prob={}'.format(self.drop_prob)



    
    

class EquivariantDropout(nn.Module):
    def __init__(self, irreps, drop_prob):
        super(EquivariantDropout, self).__init__()
        self.irreps = irreps
        self.num_irreps = irreps.num_irreps
        self.drop_prob = drop_prob
        self.drop = torch.nn.Dropout(drop_prob, True)
        self.mul = o3.ElementwiseTensorProduct(irreps, 
            o3.Irreps('{}x0e'.format(self.num_irreps)))
        
        
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        shape = (x.shape[0], self.num_irreps)
        mask = torch.ones(shape, dtype=x.dtype, device=x.device)
        mask = self.drop(mask)
        out = self.mul(x, mask)
        return out
    

class EquivariantScalarsDropout(nn.Module):
    def __init__(self, irreps, drop_prob):
        super(EquivariantScalarsDropout, self).__init__()
        self.irreps = irreps
        self.drop_prob = drop_prob
        
        
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        out = []
        start_idx = 0
        for mul, ir in self.irreps:
            temp = x.narrow(-1, start_idx, mul * ir.dim)
            start_idx += mul * ir.dim
            if ir.is_scalar():
                temp = F.dropout(temp, p=self.drop_prob, training=self.training)
            out.append(temp)
        out = torch.cat(out, dim=-1)
        return out
    
    
    def extra_repr(self):
        return 'irreps={}, drop_prob={}'.format(self.irreps, self.drop_prob)
    