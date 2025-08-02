from pickle import FALSE
from typing import Optional, List, Tuple # <--- 在这里添加 Tuple
import torch
import torch.nn as nn
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_scatter import scatter, scatter_max
from torch_scatter.composite import scatter_softmax
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn # <--- 添加或确保这一行存在
import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists
import torch_geometric
import math
from .registry import register_model
from .instance_norm import EquivariantInstanceNorm
from .graph_norm import EquivariantGraphNorm
from .layer_norm import EquivariantLayerNormV2
from .fast_layer_norm import EquivariantLayerNormFast
from .radial_func import RadialProfile
from .tensor_product_rescale import (TensorProductRescale, LinearRS,
    FullyConnectedTensorProductRescale, irreps2gate, sort_irreps_even_first)
from .fast_activation import Activation, Gate
from .drop import EquivariantDropout, EquivariantScalarsDropout, GraphDropPath, IdentityWithBatch
from .gaussian_rbf import GaussianRadialBasisLayer
from .tools import RadialBasis, ToS2Grid_block
import torch
import torch_geometric.nn as pyg_nn



_RESCALE = True
_USE_BIAS = True

_MAX_ATOM_TYPE_QM9_ORIGINAL = 10 # F has atomic number 9. Or higher if other elements present.
_MAX_ATOM_TYPE_EMBEDDING = 90 # For general NodeEmbeddingNetwork, often set high
_AVG_NUM_NODES = 18.03065905448718
_AVG_DEGREE = 15.57930850982666


def to_ptr(batch: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch vector to a pointer vector.
    This is a custom implementation to avoid dependency issues.

    Args:
        batch (torch.Tensor): A batch vector, e.g., [0, 0, 1, 2, 2, 2].

    Returns:
        torch.Tensor: A pointer vector, e.g., [0, 2, 3, 6].
    """
    if batch.numel() == 0:
        return torch.zeros(1, dtype=torch.long, device=batch.device)

    # 计算每个图的节点数
    # bincount会计算每个索引出现的次数
    counts = torch.bincount(batch)

    # 构建ptr向量
    # ptr的第一个元素总是0
    ptr = torch.cat([batch.new_zeros(1), counts.cumsum(dim=0)], dim=0)

    return ptr


def check_for_nan(tensor, name):
    """一个辅助函数，用于检查NaN并打印信息。"""
    if torch.isnan(tensor).any():
        print(f"!!!!!!!!!!!!!!! NaN DETECTED IN: {name} !!!!!!!!!!!!!!!")
        # raise ValueError(f"NaN detected in {name}") # 可以选择直接抛出异常中断

def get_norm_layer(norm_type):
    if norm_type == 'graph':
        return EquivariantGraphNorm
    elif norm_type == 'instance':
        return EquivariantInstanceNorm
    elif norm_type == 'layer':
        return EquivariantLayerNormV2
    elif norm_type == 'fast_layer':
        return EquivariantLayerNormFast
    elif norm_type is None:
        return None
    else:
        raise ValueError('Norm type {} not supported.'.format(norm_type))
    

class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.alpha = negative_slope
        
    
    def forward(self, x):
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2
    
    
    def extra_repr(self):
        return 'negative_slope={}'.format(self.alpha)
            

def get_mul_0e(irreps):
    mul_0 = 0
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            mul_0 += mul
    return mul_0


class FullyConnectedTensorProductRescaleNorm(FullyConnectedTensorProductRescale):
    
    def __init__(self, irreps_in1, irreps_in2, irreps_out,
        bias=True, rescale=True,
        internal_weights=None, shared_weights=None,
        normalization=None, norm_layer='graph'):
        
        super().__init__(irreps_in1, irreps_in2, irreps_out,
            bias=bias, rescale=rescale,
            internal_weights=internal_weights, shared_weights=shared_weights,
            normalization=normalization)
        self.norm = get_norm_layer(norm_layer)(self.irreps_out)
        
        
    def forward(self, x, y, batch, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.norm(out, batch=batch)
        return out
        

class FullyConnectedTensorProductRescaleNormSwishGate(FullyConnectedTensorProductRescaleNorm):
    
    def __init__(self, irreps_in1, irreps_in2, irreps_out,
        bias=True, rescale=True,
        internal_weights=None, shared_weights=None,
        normalization=None, norm_layer='graph'):
        
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_out)
        if irreps_gated.num_irreps == 0:
            gate = Activation(irreps_out, acts=[torch.nn.SiLU()])
        else:
            gate = Gate(
                irreps_scalars, [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                irreps_gates, [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
        super().__init__(irreps_in1, irreps_in2, gate.irreps_in,
            bias=bias, rescale=rescale,
            internal_weights=internal_weights, shared_weights=shared_weights,
            normalization=normalization, norm_layer=norm_layer)
        self.gate = gate
        
        
    def forward(self, x, y, batch, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.norm(out, batch=batch)
        out = self.gate(out)
        return out
    

class FullyConnectedTensorProductRescaleSwishGate(FullyConnectedTensorProductRescale):
    
    def __init__(self, irreps_in1, irreps_in2, irreps_out,
        bias=True, rescale=True,
        internal_weights=None, shared_weights=None,
        normalization=None):
        
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(irreps_out)
        if irreps_gated.num_irreps == 0:
            gate = Activation(irreps_out, acts=[torch.nn.SiLU()])
        else:
            gate = Gate(
                irreps_scalars, [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                irreps_gates, [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated  # gated tensors
            )
        super().__init__(irreps_in1, irreps_in2, gate.irreps_in,
            bias=bias, rescale=rescale,
            internal_weights=internal_weights, shared_weights=shared_weights,
            normalization=normalization)
        self.gate = gate
        
        
    def forward(self, x, y, weight=None):
        out = self.forward_tp_rescale_bias(x, y, weight)
        out = self.gate(out)
        return out
    

def DepthwiseTensorProduct(irreps_node_input, irreps_edge_attr, irreps_node_output, 
    internal_weights=False, bias=True):
    '''
        The irreps of output is pre-determined. 
        `irreps_node_output` is used to get certain types of vectors.
    '''
    irreps_output = []
    instructions = []
    
    for i, (mul, ir_in) in enumerate(irreps_node_input):
        for j, (_, ir_edge) in enumerate(irreps_edge_attr):
            for ir_out in ir_in * ir_edge:
                if ir_out in irreps_node_output or ir_out == o3.Irrep(0, 1):
                    k = len(irreps_output)
                    irreps_output.append((mul, ir_out))
                    instructions.append((i, j, k, 'uvu', True))
        
    irreps_output = o3.Irreps(irreps_output)
    irreps_output, p, _ = sort_irreps_even_first(irreps_output) #irreps_output.sort()
    instructions = [(i_1, i_2, p[i_out], mode, train)
        for i_1, i_2, i_out, mode, train in instructions]
    tp = TensorProductRescale(irreps_node_input, irreps_edge_attr,
            irreps_output, instructions,
            internal_weights=internal_weights,
            shared_weights=internal_weights,
            bias=bias, rescale=_RESCALE)
    return tp    


class SeparableFCTP(torch.nn.Module):
    '''
        Use separable FCTP for spatial convolution.
    '''
    def __init__(self, irreps_node_input, irreps_edge_attr, irreps_node_output, 
        fc_neurons, use_activation=False, norm_layer='graph', 
        internal_weights=False):
        
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        norm = get_norm_layer(norm_layer)
        
        self.dtp = DepthwiseTensorProduct(self.irreps_node_input, self.irreps_edge_attr, 
            self.irreps_node_output, bias=False, internal_weights=internal_weights)
        
        self.dtp_rad = None
        if fc_neurons is not None:
            self.dtp_rad = RadialProfile(fc_neurons + [self.dtp.tp.weight_numel])
            for (slice, slice_sqrt_k) in self.dtp.slices_sqrt_k.values():
                self.dtp_rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
                self.dtp_rad.offset.data[slice] *= slice_sqrt_k
                
        irreps_lin_output = self.irreps_node_output
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(self.irreps_node_output)
        if use_activation:
            irreps_lin_output = irreps_scalars + irreps_gates + irreps_gated
            irreps_lin_output = irreps_lin_output.simplify()
        self.lin = LinearRS(self.dtp.irreps_out.simplify(), irreps_lin_output)
        
        self.norm = None
        if norm_layer is not None:
            self.norm = norm(self.lin.irreps_out)

        self.gate = None
        if use_activation:
            if irreps_gated.num_irreps == 0:
                gate = Activation(self.irreps_node_output, acts=[torch.nn.SiLU()])
            else:
                gate = Gate(
                    irreps_scalars, [torch.nn.SiLU() for _, ir in irreps_scalars],  # scalar
                    irreps_gates, [torch.sigmoid for _, ir in irreps_gates],  # gates (scalars)
                    irreps_gated  # gated tensors
                )
            self.gate = gate
    
    
    def forward(self, node_input, edge_attr, edge_scalars, batch=None, **kwargs):
        '''
            Depthwise TP: `node_input` TP `edge_attr`, with TP parametrized by 
            self.dtp_rad(`edge_scalars`).
        '''
        weight = None
        if self.dtp_rad is not None and edge_scalars is not None:    
            weight = self.dtp_rad(edge_scalars)
        out = self.dtp(node_input, edge_attr, weight)
        out = self.lin(out)
        if self.norm is not None:
            out = self.norm(out, batch=batch)
        if self.gate is not None:
            out = self.gate(out)
        return out
        

@compile_mode('script')
class Vec2AttnHeads(torch.nn.Module):
    '''
        Reshape vectors of shape [N, irreps_mid] to vectors of shape
        [N, num_heads, irreps_head].
    '''
    def __init__(self, irreps_head, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.irreps_head = irreps_head
        self.irreps_mid_in = []
        for mul, ir in irreps_head:
            self.irreps_mid_in.append((mul * num_heads, ir))
        self.irreps_mid_in = o3.Irreps(self.irreps_mid_in)
        self.mid_in_indices = []
        start_idx = 0
        for mul, ir in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim
    
    
    def forward(self, x):
        N, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.mid_in_indices):
            temp = x.narrow(1, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, self.num_heads, -1)
            out.append(temp)
        out = torch.cat(out, dim=2)
        return out
    
    
    def __repr__(self):
        return '{}(irreps_head={}, num_heads={})'.format(
            self.__class__.__name__, self.irreps_head, self.num_heads)
    
    
@compile_mode('script')
class AttnHeads2Vec(torch.nn.Module):
    '''
        Convert vectors of shape [N, num_heads, irreps_head] into
        vectors of shape [N, irreps_head * num_heads].
    '''
    def __init__(self, irreps_head):
        super().__init__()
        self.irreps_head = irreps_head
        self.head_indices = []
        start_idx = 0
        for mul, ir in self.irreps_head:
            self.head_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim
    
    
    def forward(self, x):
        N, _, _ = x.shape
        out = []
        for ir_idx, (start_idx, end_idx) in enumerate(self.head_indices):
            temp = x.narrow(2, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, -1)
            out.append(temp)
        out = torch.cat(out, dim=1)
        return out
    
    
    def __repr__(self):
        return '{}(irreps_head={})'.format(self.__class__.__name__, self.irreps_head)


class ConcatIrrepsTensor(torch.nn.Module):
    
    def __init__(self, irreps_1, irreps_2):
        super().__init__()
        assert irreps_1 == irreps_1.simplify()
        self.check_sorted(irreps_1)
        assert irreps_2 == irreps_2.simplify()
        self.check_sorted(irreps_2)
        
        self.irreps_1 = irreps_1
        self.irreps_2 = irreps_2
        self.irreps_out = irreps_1 + irreps_2
        self.irreps_out, _, _ = sort_irreps_even_first(self.irreps_out) #self.irreps_out.sort()
        self.irreps_out = self.irreps_out.simplify()
        
        self.ir_mul_list = []
        lmax = max(irreps_1.lmax, irreps_2.lmax)
        irreps_max = []
        for i in range(lmax + 1):
            irreps_max.append((1, (i, -1)))
            irreps_max.append((1, (i,  1)))
        irreps_max = o3.Irreps(irreps_max)
        
        start_idx_1, start_idx_2 = 0, 0
        dim_1_list, dim_2_list = self.get_irreps_dim(irreps_1), self.get_irreps_dim(irreps_2)
        for _, ir in irreps_max:
            dim_1, dim_2 = None, None
            index_1 = self.get_ir_index(ir, irreps_1)
            index_2 = self.get_ir_index(ir, irreps_2)
            if index_1 != -1:
                dim_1 = dim_1_list[index_1]
            if index_2 != -1:
                dim_2 = dim_2_list[index_2]
            self.ir_mul_list.append((start_idx_1, dim_1, start_idx_2, dim_2))
            start_idx_1 = start_idx_1 + dim_1 if dim_1 is not None else start_idx_1
            start_idx_2 = start_idx_2 + dim_2 if dim_2 is not None else start_idx_2
          
            
    def get_irreps_dim(self, irreps):
        muls = []
        for mul, ir in irreps:
            muls.append(mul * ir.dim)
        return muls
    
    
    def check_sorted(self, irreps):
        lmax = None
        p = None
        for _, ir in irreps:
            if p is None and lmax is None:
                p = ir.p
                lmax = ir.l
                continue
            if ir.l == lmax:
                assert p < ir.p, 'Parity order error: {}'.format(irreps)
            assert lmax <= ir.l                
        
    
    def get_ir_index(self, ir, irreps):
        for index, (_, irrep) in enumerate(irreps):
            if irrep == ir:
                return index
        return -1
    
    
    def forward(self, feature_1, feature_2):
        
        output = []
        for i in range(len(self.ir_mul_list)):
            start_idx_1, mul_1, start_idx_2, mul_2 = self.ir_mul_list[i]
            if mul_1 is not None:
                output.append(feature_1.narrow(-1, start_idx_1, mul_1))
            if mul_2 is not None:
                output.append(feature_2.narrow(-1, start_idx_2, mul_2))
        output = torch.cat(output, dim=-1)
        return output
    
    
    def __repr__(self):
        return '{}(irreps_1={}, irreps_2={})'.format(self.__class__.__name__, 
            self.irreps_1, self.irreps_2)

        
@compile_mode('script')
class GraphAttention(torch.nn.Module):
    '''
        1. Message = Alpha * Value
        2. Two Linear to merge src and dst -> Separable FCTP -> 0e + (0e+1e+...)
        3. 0e -> Activation -> Inner Product -> (Alpha)
        4. (0e+1e+...) -> (Value)
    '''
    def __init__(self,
        irreps_node_input, irreps_node_attr,
        irreps_edge_attr, irreps_node_output,
        fc_neurons,
        irreps_head, num_heads, irreps_pre_attn=None, 
        rescale_degree=False, nonlinear_message=False,
        alpha_drop=0.1, proj_drop=0.1):
        
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = self.irreps_node_input if irreps_pre_attn is None \
            else o3.Irreps(irreps_pre_attn)
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        
        # Merge src and dst
        self.merge_src = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=True)
        self.merge_dst = LinearRS(self.irreps_node_input, self.irreps_pre_attn, bias=False)
        
        irreps_attn_heads = irreps_head * num_heads
        irreps_attn_heads, _, _ = sort_irreps_even_first(irreps_attn_heads) #irreps_attn_heads.sort()
        irreps_attn_heads = irreps_attn_heads.simplify() 
        mul_alpha = get_mul_0e(irreps_attn_heads)
        mul_alpha_head = mul_alpha // num_heads
        irreps_alpha = o3.Irreps('{}x0e'.format(mul_alpha)) # for attention score
        irreps_attn_all = (irreps_alpha + irreps_attn_heads).simplify()
        
        self.sep_act = None
        if self.nonlinear_message:
            # Use an extra separable FCTP and Swish Gate for value
            self.sep_act = SeparableFCTP(self.irreps_pre_attn, 
                self.irreps_edge_attr, self.irreps_pre_attn, fc_neurons, 
                use_activation=True, norm_layer=None, internal_weights=False)
            self.sep_alpha = LinearRS(self.sep_act.dtp.irreps_out, irreps_alpha)
            self.sep_value = SeparableFCTP(self.irreps_pre_attn, 
                self.irreps_edge_attr, irreps_attn_heads, fc_neurons=None, 
                use_activation=False, norm_layer=None, internal_weights=True)
            self.vec2heads_alpha = Vec2AttnHeads(o3.Irreps('{}x0e'.format(mul_alpha_head)), 
                num_heads)
            self.vec2heads_value = Vec2AttnHeads(self.irreps_head, num_heads)
        else:
            self.sep = SeparableFCTP(self.irreps_pre_attn, 
                self.irreps_edge_attr, irreps_attn_all, fc_neurons, 
                use_activation=False, norm_layer=None)
            self.vec2heads = Vec2AttnHeads(
                (o3.Irreps('{}x0e'.format(mul_alpha_head)) + irreps_head).simplify(), 
                num_heads)
        
        self.alpha_act = Activation(o3.Irreps('{}x0e'.format(mul_alpha_head)), 
            [SmoothLeakyReLU(0.2)])
        self.heads2vec = AttnHeads2Vec(irreps_head)
        
        self.mul_alpha_head = mul_alpha_head
        self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        torch_geometric.nn.inits.glorot(self.alpha_dot) # Following GATv2
        
        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)
        
        self.proj = LinearRS(irreps_attn_heads, self.irreps_node_output)
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(self.irreps_node_input, 
                drop_prob=proj_drop)
        
        
    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars, 
        batch, **kwargs):
        
        message_src = self.merge_src(node_input)
        message_dst = self.merge_dst(node_input)
        message = message_src[edge_src] + message_dst[edge_dst]
        
        if self.nonlinear_message:
            weight = self.sep_act.dtp_rad(edge_scalars)
            message = self.sep_act.dtp(message, edge_attr, weight)
            alpha = self.sep_alpha(message)
            alpha = self.vec2heads_alpha(alpha)
            value = self.sep_act.lin(message)
            value = self.sep_act.gate(value)
            value = self.sep_value(value, edge_attr=edge_attr, edge_scalars=edge_scalars)
            value = self.vec2heads_value(value)
        else:
            message = self.sep(message, edge_attr=edge_attr, edge_scalars=edge_scalars)
            message = self.vec2heads(message)
            head_dim_size = message.shape[-1]
            alpha = message.narrow(2, 0, self.mul_alpha_head)
            value = message.narrow(2, self.mul_alpha_head, (head_dim_size - self.mul_alpha_head))
        
        # inner product
        alpha = self.alpha_act(alpha)
        alpha = torch.einsum('bik, aik -> bi', alpha, self.alpha_dot)
        alpha = torch_geometric.utils.softmax(alpha, edge_dst)
        alpha = alpha.unsqueeze(-1)
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)
        attn = value * alpha
        attn = scatter(attn, index=edge_dst, dim=0, dim_size=node_input.shape[0])
        attn = self.heads2vec(attn)
        
        if self.rescale_degree:
            degree = torch_geometric.utils.degree(edge_dst, 
                num_nodes=node_input.shape[0], dtype=node_input.dtype)
            degree = degree.view(-1, 1)
            attn = attn * degree
            
        node_output = self.proj(attn)
        
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        
        return node_output
    
    
    def extra_repr(self):
        output_str = super(GraphAttention, self).extra_repr()
        output_str = output_str + 'rescale_degree={}, '.format(self.rescale_degree)
        return output_str
                    

@compile_mode('script')
class FeedForwardNetwork(torch.nn.Module):
    '''
        Use two (FCTP + Gate)
    '''
    def __init__(self,
        irreps_node_input, irreps_node_attr,
        irreps_node_output, irreps_mlp_mid=None,
        proj_drop=0.1):
        
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid) if irreps_mlp_mid is not None \
            else self.irreps_node_input
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        
        self.fctp_1 = FullyConnectedTensorProductRescaleSwishGate(
            self.irreps_node_input, self.irreps_node_attr, self.irreps_mlp_mid, 
            bias=True, rescale=_RESCALE)
        self.fctp_2 = FullyConnectedTensorProductRescale(
            self.irreps_mlp_mid, self.irreps_node_attr, self.irreps_node_output, 
            bias=True, rescale=_RESCALE)
        
        self.proj_drop = None
        if proj_drop != 0.0:
            self.proj_drop = EquivariantDropout(self.irreps_node_output, 
                drop_prob=proj_drop)
            
        
    def forward(self, node_input, node_attr, **kwargs):
        node_output = self.fctp_1(node_input, node_attr)
        node_output = self.fctp_2(node_output, node_attr)
        if self.proj_drop is not None:
            node_output = self.proj_drop(node_output)
        return node_output
    

class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -3.0,
        stop: float = 3.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:

        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

#@compile_mode('script')
class TransBlock(torch.nn.Module):
    '''
        1. Layer Norm 1 -> GraphAttention -> Layer Norm 2 -> FeedForwardNetwork
        2. Use pre-norm architecture
    '''

    def __init__(self,
                 irreps_node_input: str,  # e.g., '128x0e+64x1e+32x2e'
                 irreps_node_attr: str,  # e.g., '1x0e' (dummy scalar attribute)
                 irreps_edge_attr: str,  # e.g., '1x0e+1x1e+1x2e' (spherical harmonics for edges)
                 irreps_node_output: str,  # Usually same as irreps_node_input for residual blocks
                 fc_neurons: List[int],  # For RadialProfile in GraphAttention's SeparableFCTP e.g. [64, 64]
                 irreps_head: str,  # For GraphAttention, e.g., '32x0e+16x1e+8x2e'
                 num_heads: int,  # For GraphAttention
                 irreps_pre_attn=None,  # For GraphAttention
                 rescale_degree: bool = False,  # For GraphAttention
                 nonlinear_message: bool = False,  # For GraphAttention
                 alpha_drop: float = 0.1,  # For GraphAttention
                 proj_drop: float = 0.1,  # For GraphAttention and FFN
                 drop_path_rate: float = 0.0,
                 irreps_mlp_mid=None,  # For FFN hidden layer
                 norm_layer='layer',
                 # SSP related parameters
                 ssp_module: bool = False,  # Flag to enable SSP specific layers/logic
                 ssp_feature_dim: str = '32x0e',  # Irreps of the combined (prop + masked_atom_type) SSP feature
                 # This is the input to fc_ssp_gate_scalar and fc_ssp_value_transform
                 # This defines the irreps of `aggregated_masked_atom_info_per_graph`
                 # and the irreps of `prop_embedded` if `prop_value_per_graph` is used.
                 # These two are combined to form `ssp_info_per_graph` which should match `ssp_feature_dim`.
                 # For simplicity, we assume ssp_feature_dim will be the Irreps of the combined feature.
                 ):
        super().__init__()
        self.ssp_module = ssp_module

        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)

        # Layer Normalizations
        self.norm_1 = get_norm_layer(norm_layer)(self.irreps_node_input)
        self.norm_2 = get_norm_layer(norm_layer)(self.irreps_node_input)  # Input to FFN is output of GA block

        # Graph Attention
        _irreps_pre_attn = self.irreps_node_input if irreps_pre_attn is None else o3.Irreps(irreps_pre_attn)
        _irreps_ga_output = self.irreps_node_input  # GA output for residual connection
        self.ga = GraphAttention(
            irreps_node_input=self.irreps_node_input,  # Input to GA is after norm1
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr,
            irreps_node_output=_irreps_ga_output,
            fc_neurons=fc_neurons,
            irreps_head=o3.Irreps(irreps_head),
            num_heads=num_heads,
            irreps_pre_attn=_irreps_pre_attn,
            rescale_degree=rescale_degree,
            nonlinear_message=nonlinear_message,
            alpha_drop=alpha_drop,
            proj_drop=proj_drop
        )

        # DropPath
        self.drop_path = GraphDropPath(drop_path_rate) if drop_path_rate > 0. else IdentityWithBatch()

        # FeedForward Network
        _irreps_mlp_mid_obj = self.irreps_node_input if irreps_mlp_mid is None else o3.Irreps(irreps_mlp_mid)
        self.ffn = FeedForwardNetwork(
            irreps_node_input=self.irreps_node_input,  # Input to FFN is after norm2
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_node_output,  # FFN can change output irreps for the block
            irreps_mlp_mid=_irreps_mlp_mid_obj,
            proj_drop=proj_drop
        )

        # Shortcut connection for FFN (if input and output irreps of the block differ)
        self.ffn_shortcut = nn.Identity()
        # The shortcut should connect the INPUT of the FFN part to the OUTPUT of the FFN part
        # Input to FFN part is output of GA part (after residual).
        # Output of FFN part is self.irreps_node_output.
        # So, if self.irreps_node_input (output of GA part for residual) != self.irreps_node_output (output of FFN)
        if self.irreps_node_input != self.irreps_node_output:
            self.ffn_shortcut = LinearRS(self.irreps_node_input, self.irreps_node_output)

        # --- SSP Specific Layers (only if ssp_module is True) ---
        if self.ssp_module:
            self.ssp_feature_dim_irreps = o3.Irreps(ssp_feature_dim)  # e.g., o3.Irreps('32x0e')

            # --- Configure self.prop_embed to output 16x0e ---
            # This dimension should be chosen such that when concatenated with
            # agg_atom_embed_from_target's embedding, it matches self.ssp_feature_dim_irreps
            # If ssp_feature_dim is 32x0e, and agg_atom_embed is 16x0e, then prop_embed should be 16x0e.
            self.prop_embed_out_dim = 16  # Explicitly set for this design
            self.prop_embed_irreps = o3.Irreps(f"{self.prop_embed_out_dim}x0e")  # '16x0e'

            if self.prop_embed_out_dim > 0:  # Only create if needed
                self.prop_embed = nn.Sequential(
                    # GaussianSmearing input is (B,), output (B, num_gaussians)
                    GaussianSmearing(start=-5.0, stop=5.0, num_gaussians=self.prop_embed_out_dim * 2),
                    # Example: output (B, 32)
                    nn.Linear(self.prop_embed_out_dim * 2, self.prop_embed_out_dim)  # Output (B, 16)
                )
            else:
                self.prop_embed = None  # No property embedding if out_dim is 0


            # Activation for the combined SSP feature
            self.act_ssp_embed = nn.SiLU()

            # Gate scalar MLP: ssp_feature_dim_irreps -> 1x0e -> Sigmoid
            gate_scalar_hidden_dim = max(1, get_mul_0e(self.ssp_feature_dim_irreps) // 2)  # e.g., 16 if input is 32x0e
            self.fc_ssp_gate_scalar = nn.Sequential(
                LinearRS(self.ssp_feature_dim_irreps, o3.Irreps(f'{gate_scalar_hidden_dim}x0e')),
                nn.SiLU(),
                LinearRS(o3.Irreps(f'{gate_scalar_hidden_dim}x0e'), o3.Irreps('1x0e')),
                nn.Sigmoid()
            )

            # Value transform: ssp_feature_dim_irreps -> irreps_node_input
            self.fc_ssp_value_transform = LinearRS(
                self.ssp_feature_dim_irreps,
                self.irreps_node_input,  # Output matches node_input for element-wise product & sum
                bias=False  # Often, the value transform has no bias if it's added later
            )

    def forward(self, node_input: torch.Tensor,
                node_attr: torch.Tensor,  # Scalar attributes for FFN, GA
                edge_src: torch.Tensor, edge_dst: torch.Tensor,
                edge_attr: torch.Tensor,  # SH for edges
                edge_scalars: torch.Tensor,  # Radial basis for edges
                batch: torch.Tensor,
                target: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                # For SSP: (prop_value_per_graph_embed, aggregated_masked_atom_info_per_graph_embed)
                ) -> torch.Tensor:

        node_input_for_first_residual = node_input  # Save for the first residual connection

        # --- SSP Feature Injection with Gating (if enabled and target provided) ---
        if self.ssp_module and target is not None:
            raw_prop_value_per_graph, agg_atom_embed_from_target = target
            # raw_prop_value_per_graph is 1D: (num_graphs,)
            # agg_atom_embed_from_target is 2D: (num_graphs, 16) due to atom_embed_ssp_injection config

            tensors_to_combine = []
            current_prop_embedded = None

            if raw_prop_value_per_graph is not None:
                if self.prop_embed is not None:  # prop_embed configured to output (num_graphs, 16)
                    prop_input_for_smearing = raw_prop_value_per_graph.float()
                    current_prop_embedded = self.prop_embed(prop_input_for_smearing)  # Shape: (num_graphs, 16)
                    if current_prop_embedded.ndim != 2:
                        raise ValueError(f"Embedded property is not 2D, shape: {current_prop_embedded.shape}")
                    tensors_to_combine.append(current_prop_embedded)
                # else: Property value provided but no embedder defined in TransBlock. This might be an issue.

            if agg_atom_embed_from_target is not None:
                if agg_atom_embed_from_target.ndim != 2:
                    raise ValueError(f"Aggregated atom embed is not 2D, shape: {agg_atom_embed_from_target.shape}")
                tensors_to_combine.append(agg_atom_embed_from_target)  # Shape: (num_graphs, 16)

            if not tensors_to_combine:
                # ... (handle empty list as before) ...
                if self.ssp_feature_dim_irreps.dim > 0:
                    raise ValueError("SSP: No valid features in tensors_to_combine to form ssp_info_per_graph...")
                # ... create empty ssp_info_per_graph ...
            elif len(tensors_to_combine) == 1:
                ssp_info_per_graph = tensors_to_combine[0]  # Should have dim 16 (either prop or atom)
            else:  # Both are present, concatenate them
                ssp_info_per_graph = torch.cat(tensors_to_combine, dim=-1)  # (B, 16+16=32)

            # Dimension check after combination
            if ssp_info_per_graph.shape[-1] != self.ssp_feature_dim_irreps.dim:  # Expected 32 vs actual
                print(
                    f"DEBUG TransBlock current_prop_embedded shape: {current_prop_embedded.shape if current_prop_embedded is not None else 'None'}")
                print(
                    f"DEBUG TransBlock agg_atom_embed_from_target shape: {agg_atom_embed_from_target.shape if agg_atom_embed_from_target is not None else 'None'}")
                print(f"DEBUG TransBlock ssp_info_per_graph after combination shape: {ssp_info_per_graph.shape}")
                print(
                    f"DEBUG TransBlock self.ssp_feature_dim_irreps: {self.ssp_feature_dim_irreps}, expected final dim: {self.ssp_feature_dim_irreps.dim}")
                raise ValueError(f"SSP Info Dim Mismatch...")

            ssp_info_per_graph_activated = self.act_ssp_embed(ssp_info_per_graph)

            gate_signal_per_graph = self.fc_ssp_gate_scalar(ssp_info_per_graph_activated)
            ssp_value_transformed_per_graph = self.fc_ssp_value_transform(ssp_info_per_graph_activated)

            gate_signal_expanded = torch.index_select(gate_signal_per_graph, 0, batch)
            ssp_value_expanded = torch.index_select(ssp_value_transformed_per_graph, 0, batch)

            node_input = node_input + gate_signal_expanded * ssp_value_expanded
        # --- End SSP Feature Injection ---

        node_input_for_first_residual = node_input

        # --- 1. Attention sub-block ---
        check_for_nan(node_input, "TransBlock input")
        x_norm1 = self.norm_1(node_input, batch=batch)
        check_for_nan(x_norm1, "TransBlock after norm_1")

        x_attn = self.ga(
            node_input=x_norm1, node_attr=node_attr,
            edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_attr,
            edge_scalars=edge_scalars, batch=batch
        )
        check_for_nan(x_attn, "TransBlock after GraphAttention")

        x_after_attn_res = node_input_for_first_residual + self.drop_path(x_attn, batch=batch)
        check_for_nan(x_after_attn_res, "TransBlock after first residual")

        # --- 2. FFN sub-block ---
        x_norm2 = self.norm_2(x_after_attn_res, batch=batch)
        check_for_nan(x_norm2, "TransBlock after norm_2")

        x_ffn = self.ffn(
            node_input=x_norm2, node_attr=node_attr
        )
        check_for_nan(x_ffn, "TransBlock after FFN")

        shortcut_path_for_ffn = self.ffn_shortcut(x_after_attn_res)

        node_output = shortcut_path_for_ffn + self.drop_path(x_ffn, batch=batch)
        check_for_nan(node_output, "TransBlock final output")

        return node_output

class ExpNormalSmearing(torch.nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=False):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )

class TargetEdgeEmbedding(torch.nn.Module):
    def __init__(self, irreps_node_embedding, irreps_edge_attr, fc_neurons, basis_type):
        super().__init__()
        self.basis_type = basis_type
        self.irreps_node_embedding = irreps_node_embedding
        self.exp = LinearRS(o3.Irreps('1x0e'), irreps_node_embedding, 
            bias=_USE_BIAS, rescale=_RESCALE)
        self.dw = DepthwiseTensorProduct(irreps_node_embedding, 
            irreps_edge_attr, irreps_node_embedding, 
            internal_weights=False, bias=False)
        self.rad = RadialProfile(fc_neurons + [self.dw.tp.weight_numel])
        for (slice, slice_sqrt_k) in self.dw.slices_sqrt_k.values():
            self.rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
            self.rad.offset.data[slice] *= slice_sqrt_k
        self.proj = LinearRS(self.dw.irreps_out.simplify(), irreps_node_embedding)

        self.rbf = GaussianRadialBasisLayer(32, cutoff=5.0)
        
    
    def forward(self, x):
        weight = self.rbf(x.norm(dim=1))
        weight = self.rad(weight)
        features = o3.spherical_harmonics(l=torch.arange(self.irreps_node_embedding.lmax + 1).tolist(), x=x, normalize=True, normalization='component')
        # features = self.proj(features)
        one_features = torch.ones_like(x.narrow(1, 0, 1))
        one_features = self.exp(one_features)
        features = self.dw(one_features, features, weight)
        features = self.proj(features)

        return features
        
class NodeEmbeddingNetwork(torch.nn.Module):
    
    def __init__(self, irreps_node_embedding, max_atom_type=_MAX_ATOM_TYPE_EMBEDDING, bias=True): # Use general max
        super().__init__()
        self.max_atom_type_internal = max_atom_type # Max index for one-hot
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.atom_type_lin = LinearRS(
            o3.Irreps('{}x0e'.format(self.max_atom_type_internal)),
            self.irreps_node_embedding,
            bias=bias
        )
        # Scaling factor can be sqrt of (max_atom_type_internal) or number of actual types if known
        self.atom_type_lin.tp.weight.data.mul_(self.max_atom_type_internal ** 0.5)

    def forward(self, node_atom_indices): # Expects indices from 0 to max_atom_type_internal-1
        # Ensure indices are within expected range. Clamp or error if out of bounds.
        node_atom_indices_clamped = torch.clamp(node_atom_indices, 0, self.max_atom_type_internal - 1)
        if not torch.all(node_atom_indices == node_atom_indices_clamped):
            # print(f"Warning: Atom indices clamped in NodeEmbeddingNetwork. Original: {node_atom_indices.unique()}, Max expected: {self.max_atom_type_internal-1}")
            pass # Or raise error

        node_atom_onehot = torch.nn.functional.one_hot(node_atom_indices_clamped, self.max_atom_type_internal).float()
        # node_attr is just the one-hot for this simple embedding
        node_attr_onehot = node_atom_onehot
        node_embedding = self.atom_type_lin(node_atom_onehot)
        return node_embedding, node_attr_onehot, node_atom_onehot # Returning onehot for attr and explicit onehot


class ScaledScatter(torch.nn.Module):
    def __init__(self, avg_aggregate_num):
        super().__init__()
        self.avg_aggregate_num = avg_aggregate_num + 0.0


    def forward(self, x, index, **kwargs):
        out = scatter(x, index, **kwargs)
        out = out.div(self.avg_aggregate_num ** 0.5)
        return out
    
    
    def extra_repr(self):
        return 'avg_aggregate_num={}'.format(self.avg_aggregate_num)
    

class EdgeDegreeEmbeddingNetwork(torch.nn.Module):

    def __init__(self, irreps_node_embedding, irreps_edge_attr, fc_neurons, avg_aggregate_num):
        super().__init__()
        self.exp = LinearRS(o3.Irreps('1x0e'), irreps_node_embedding,
                            bias=_USE_BIAS, rescale=_RESCALE)
        self.dw = DepthwiseTensorProduct(irreps_node_embedding,
                                         irreps_edge_attr, irreps_node_embedding,
                                         internal_weights=False, bias=False)
        self.rad = RadialProfile(fc_neurons + [self.dw.tp.weight_numel])
        for (slice, slice_sqrt_k) in self.dw.slices_sqrt_k.values():
            self.rad.net[-1].weight.data[slice, :] *= slice_sqrt_k
            self.rad.offset.data[slice] *= slice_sqrt_k
        self.proj = LinearRS(self.dw.irreps_out.simplify(), irreps_node_embedding)
        self.scale_scatter = ScaledScatter(avg_aggregate_num)

    def forward(self, node_input, edge_attr, edge_scalars, edge_src, edge_dst, batch):
        node_features = torch.ones_like(node_input.narrow(1, 0, 1))
        node_features = self.exp(node_features)
        weight = self.rad(edge_scalars)
        edge_features = self.dw(node_features[edge_src], edge_attr, weight)
        edge_features = self.proj(edge_features)
        node_features = self.scale_scatter(edge_features, edge_dst, dim=0,
                                           dim_size=node_features.shape[0])
        return node_features
    


import torch_geometric.nn as pyg_nn
class GraphAttentionTransformer(torch.nn.Module):
    def __init__(self,
                 # 核心模型架构参数
                 irreps_in: str,
                 irreps_node_embedding: str = '128x0e+64x1e+32x2e',
                 num_layers: int = 6,
                 irreps_node_attr: str = '1x0e',
                 task_type='qm9',
                 irreps_sh: str = '1x0e+1x1e+1x2e',
                 max_radius: float = 5.0,
                 number_of_basis: int = 128,
                 fc_neurons: list = [64, 64],
                 basis_type: str = 'gaussian',
                 irreps_feature: str = '512x0e',
                 irreps_head: str = '32x0e+16x1e+8x2e',
                 num_heads: int = 4,
                 nonlinear_message: bool = True,
                 irreps_mlp_mid: str = '384x0e+192x1e+96x2e',
                 norm_layer: str = 'layer',

                 # 正则化和Dropout参数
                 alpha_drop: float = 0.2,
                 proj_drop: float = 0.0,
                 out_drop: float = 0.0,
                 drop_path_rate: float = 0.0,

                 # 主任务相关参数 (来自你的原始代码)
                 task_mean=None,
                 task_std=None,
                 atomref=None,  # 虽然未使用，但保留接口
                 scale=None,  # 虽然未使用，但保留接口

                 # EMPP (SSP) 辅助任务参数
                 ssp: bool = False,
                 empp_num_mask: int = 1,
                 empp_ssp_feature_dim: str = '16x0e',
                 empp_atom_type_embed_irreps: str = '16x0e',
                 empp_prioritize_heteroatoms: bool = True,
                 # pos_prediction head 的参数
                 empp_pos_pred_num_s2_channels: int = 32,
                 empp_pos_pred_res_s2grid: int = 100,
                 empp_pos_pred_temp_softmax: float = 0.1,
                 empp_pos_pred_temp_label: float = 0.1,

                 # 对比学习辅助任务参数
                 enable_contrastive_learning: bool = False,
                 contrastive_projection_dim: int = 128,

                 # 原子映射信息 (硬编码或从配置加载)
                 # 你可以从 model_entrypoint 函数中传递这些固定的字典/列表
                 qm9_atom_type_map: dict = None,
                 heteroatom_mapped_indices: list = None,

                 **kwargs):  # 捕获任何未使用的参数
        super().__init__()

        # --- 存储关键配置 ---
        self.task_type = task_type
        self.ssp_enabled = ssp
        self.enable_contrastive_learning = enable_contrastive_learning
        self.num_mask = empp_num_mask
        self.prioritize_heteroatoms_in_empp = empp_prioritize_heteroatoms  # 存储EMPP的策略
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis  # <--- 关键赋值！
        self.basis_type = basis_type
        if self.basis_type == 'gaussian':
            self.rbf = GaussianRadialBasisLayer(self.number_of_basis, cutoff=self.max_radius)
        elif self.basis_type == 'bessel':
            self.rbf = RadialBasis(self.number_of_basis, cutoff=self.max_radius,
                                   rbf={'name': 'spherical_bessel'})
        elif self.basis_type == 'exp':
            self.rbf = ExpNormalSmearing(cutoff_lower=0.0, cutoff_upper=self.max_radius,
                num_rbf=self.number_of_basis, trainable=False)
        else:
            raise ValueError

        # --- 统一管理原子映射信息 ---
        # 使用 register_buffer 将它们变为模型状态的一部分，并自动移动到GPU
        if qm9_atom_type_map:
            max_z = max(k for k in qm9_atom_type_map.keys() if k >= 0)
            map_list = [-1] * (max_z + 1)
            for z, internal_idx in qm9_atom_type_map.items():
                if z >= 0:
                    map_list[z] = internal_idx
            self.register_buffer('internal_atom_map_tensor', torch.tensor(map_list, dtype=torch.long))
            self.max_internal_atom_idx = max(map_list) + 1
        else:
            # Fallback if no map is provided
            self.register_buffer('internal_atom_map_tensor', torch.arange(100, dtype=torch.long))
            self.max_internal_atom_idx = 100

        if heteroatom_mapped_indices:
            self.register_buffer('heteroatom_internal_indices',
                                 torch.tensor(heteroatom_mapped_indices, dtype=torch.long))
        else:
            self.register_buffer('heteroatom_internal_indices', torch.tensor([], dtype=torch.long))

        # --- 定义 Irreps 对象 ---
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax
        self.irreps_edge_attr = o3.Irreps(irreps_sh)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_feature_before_head = o3.Irreps(irreps_feature)

        # --- 初始化嵌入层 ---
        # 主GNN使用的原子嵌入网络
        self.atom_embed_main = NodeEmbeddingNetwork(
            self.irreps_node_embedding, max_atom_type=self.max_internal_atom_idx
        )
        # EMPP任务用于特征注入的原子类型嵌入网络
        if ssp:
            self.atom_embed_ssp_injection = NodeEmbeddingNetwork(
                empp_atom_type_embed_irreps, max_atom_type=self.max_internal_atom_idx
            )


        # 边度数嵌入 (结合了初始节点特征和邻居信息)
        _AVG_DEGREE = 15.5  # Example value for QM9, can be a parameter
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(
            self.irreps_node_embedding, self.irreps_edge_attr, [number_of_basis] + fc_neurons, _AVG_DEGREE
        )

        # --- 新增: 速度嵌入层 ---
        if self.task_type == 'md17':
            # 找到 irreps_node_embedding 中 l=1 的部分
            l1_irreps = o3.Irreps([(mul, ir) for mul, ir in self.irreps_node_embedding if ir.l == 1])
            if l1_irreps.dim == 0:
                raise ValueError("irreps_node_embedding must contain l=1 components for MD17 task to embed velocity.")
            self.vel_embed = LinearRS(o3.Irreps('1x1e'), l1_irreps, bias=False)

        # ======================== 在这里添加缺失的代码 ========================
        # Pre-compute and cache the slice for l=1 features to improve efficiency
        # This slice will be used in _run_gnn_backbone to inject velocity features.
        self.l1_slice = None
        # Get a list of slice objects for each irrep component
        slices = self.irreps_node_embedding.slices()
        # Iterate through the irreps and their corresponding slices
        for ir, s in zip(self.irreps_node_embedding, slices):
            # Check if the irrep is of order l=1
            if ir.ir.l == 1:
                self.l1_slice = s  # Store the slice object
                break  # Found it, no need to continue the loop
        # =====================================================================

        self.rbf = GaussianRadialBasisLayer(number_of_basis, cutoff=max_radius)
        _AVG_DEGREE = 15.5
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(self.irreps_node_embedding, self.irreps_edge_attr,
                                                         [number_of_basis] + fc_neurons, _AVG_DEGREE)
        # --- GNN 主干网络 ---
        self.blocks = nn.ModuleList([
            TransBlock(
                irreps_node_input=self.irreps_node_embedding, irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr, irreps_node_output=self.irreps_node_embedding,
                fc_neurons=[number_of_basis] + fc_neurons, irreps_head=irreps_head,
                num_heads=num_heads, nonlinear_message=nonlinear_message, alpha_drop=alpha_drop,
                proj_drop=proj_drop, drop_path_rate=dpr, norm_layer=norm_layer,
                irreps_mlp_mid=irreps_mlp_mid, ssp_module=ssp, ssp_feature_dim=empp_ssp_feature_dim
            ) for dpr in [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        ])
            # ==============================================================

        # --- 主任务预测分支 ---
        self.fc_inv_after_blocks = LinearRS(self.irreps_node_embedding, self.irreps_feature_before_head)
        self.norm_after_blocks = get_norm_layer(norm_layer)(self.irreps_feature_before_head)
        self.out_dropout_layer = nn.Dropout(out_drop) if out_drop > 0.0 else nn.Identity()

        # --- 关键修改: 根据任务类型定义输出头 ---
        if self.task_type == 'qm9':
            self.scale_scatter_output = pyg_nn.global_mean_pool
            self.main_task_head = nn.Sequential(
                LinearRS(self.irreps_feature_before_head, self.irreps_feature_before_head, bias=True),
                Activation(self.irreps_feature_before_head, acts=[nn.SiLU()]),
                LinearRS(self.irreps_feature_before_head, o3.Irreps('1x0e'), bias=True))
        elif self.task_type == 'md17':
            self.main_task_head = LinearRS(self.irreps_node_embedding, o3.Irreps('1x1e'), bias=False)
            self.main_task_head.tp.weight.data.fill_(0.0)  # Start with predicting zero displacement

        # --- EMPP 任务分支 ---
        if self.ssp_enabled:
            # 这个pos_prediction模块的定义需要根据你的实现来确定
            # 它需要接收邻居节点的特征，并输出KL散度损失
            self.pos_prediction_module = pos_prediction(
                irreps_input_to_pos_pred=self.irreps_node_embedding,
                norm_for_pos_pred=get_norm_layer(norm_layer)(self.irreps_node_embedding),
                # `pos_prediction`也需要知道原子类型嵌入的irreps，以便处理输入
                base_atom_embedding_irreps=empp_atom_type_embed_irreps,
                lmax_for_s2_basis=self.lmax,
                num_s2_channels=empp_pos_pred_num_s2_channels,
                res_s2grid=empp_pos_pred_res_s2grid,
                temperature_softmax=empp_pos_pred_temp_softmax,
                temperature_label_softmax=empp_pos_pred_temp_label
            )

        # --- 对比学习任务分支 ---
        if self.enable_contrastive_learning:
            self.graph_pool_contrastive = pyg_nn.global_mean_pool
            gnn_output_feature_dim = self.irreps_feature_before_head.dim

            # 一个简单的两层MLP作为投影头
            self.projection_head_contrastive = nn.Sequential(
                nn.Linear(gnn_output_feature_dim, gnn_output_feature_dim),
                nn.SiLU(),  # 或者 ReLU
                nn.Linear(gnn_output_feature_dim, contrastive_projection_dim)
            )

        # 可选：应用权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):  # Catches PyTorch LayerNorm
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
        # Add inits for custom e3nn norm layers if needed, though they often handle their own.

    @torch.jit.ignore
    def no_weight_decay(self):
        # Standard no_weight_decay logic from timm or similar
        no_wd_list = []
        # ... (implementation for collecting bias and norm layer weights)
        return set(no_wd_list)

    def _map_atom_types_to_internal(self, node_atom_raw_z):

        map_tensor = self.internal_atom_map_tensor.to(node_atom_raw_z.device)


        map_tensor = self.internal_atom_map_tensor.to(node_atom_raw_z.device)

        # 对输入的原子序数(Z值)进行范围限制，防止索引越界
        # 比如数据集中出现了比我们map里定义的最大Z值还大的原子
        clamped_z = torch.clamp(node_atom_raw_z, 0, len(map_tensor) - 1)

        # 使用修正后的张量进行索引，得到内部索引
        mapped_node_atom = map_tensor[clamped_z]

        # 检查是否有未被映射的原子（在我们的map_tensor中是-1）
        # 如果有，可以打印警告或者将其映射到一个“未知”类别
        # （这里为了简单，暂时不处理，你的NodeEmbeddingNetwork可能会clamp(0)处理掉）
        if (mapped_node_atom == -1).any():
            # 这里可以加一个警告，但暂时不影响运行
            pass

        return mapped_node_atom

    def mask_position(self, pos, vel,batch_all_atoms, node_atom_all_atoms_raw_z, ptr_all_atoms,
                      main_task_target_per_graph):

        #print("--- [DEBUG-M1] Entering mask_position... ---", flush=True)
        node_atom_all_atoms_internal_idx = self._map_atom_types_to_internal(node_atom_all_atoms_raw_z)

        num_molecules = len(ptr_all_atoms) - 1
        epsilon_for_weights = 0.1  # To avoid division by zero and give some chance to frequent atoms

        # List to collect the global indices of atoms selected for masking from each molecule
        final_masked_indices_for_batch_list = []

        for i in range(num_molecules):
            mol_start_idx = ptr_all_atoms[i]
            mol_end_idx = ptr_all_atoms[i + 1]

            # Data for the current molecule
            mol_global_indices = torch.arange(mol_start_idx, mol_end_idx, device=pos.device)
            mol_atom_internal_idx_types = node_atom_all_atoms_internal_idx[mol_start_idx:mol_end_idx]

            if len(mol_global_indices) == 0:  # Skip empty molecules if any
                continue

            # Identify heteroatoms and non-heteroatoms in this molecule
            is_hetero_in_mol_mask = torch.isin(
                mol_atom_internal_idx_types,
                self.heteroatom_internal_indices.to(mol_atom_internal_idx_types.device)
            )

            hetero_global_indices_in_mol = mol_global_indices[is_hetero_in_mol_mask]
            hetero_types_in_mol = mol_atom_internal_idx_types[is_hetero_in_mol_mask]

            non_hetero_global_indices_in_mol = mol_global_indices[~is_hetero_in_mol_mask]

            # Atoms selected for masking from this specific molecule
            selected_atoms_for_this_mol_list = []
            num_to_mask_for_this_mol = self.num_mask  # How many atoms to try to mask for this molecule

            # --- Step 1: Weighted sampling from heteroatoms ---
            if len(hetero_global_indices_in_mol) > 0 and num_to_mask_for_this_mol > 0:
                # Calculate counts of each unique heteroatom type within this molecule
                unique_h_types, counts_per_h_type = torch.unique(hetero_types_in_mol, return_counts=True)
                type_to_count_map = {
                    utype.item(): count.item() for utype, count in zip(unique_h_types, counts_per_h_type)
                }

                # Assign weights to each heteroatom instance (inverse of its type's count in this molecule)
                weights_for_hetero_instances = torch.tensor(
                    [1.0 / (type_to_count_map.get(htype.item(), 1) + epsilon_for_weights) for htype in
                     hetero_types_in_mol],
                    dtype=torch.float, device=pos.device
                )

                # Ensure weights are positive if all counts were huge making 1/(count+eps) ~0
                if not torch.any(weights_for_hetero_instances > 1e-9):  # Check if all weights are effectively zero
                    weights_for_hetero_instances.add_(1e-6)

                num_samples_from_hetero = min(num_to_mask_for_this_mol, len(hetero_global_indices_in_mol))

                if num_samples_from_hetero > 0:
                    try:
                        sampled_indices_within_hetero_list = torch.multinomial(
                            weights_for_hetero_instances,
                            num_samples=num_samples_from_hetero,
                            replacement=False  # No replacement
                        )
                        selected_hetero_indices = hetero_global_indices_in_mol[sampled_indices_within_hetero_list]
                        selected_atoms_for_this_mol_list.append(selected_hetero_indices)
                        num_to_mask_for_this_mol -= len(selected_hetero_indices)
                    except RuntimeError as e:
                        # multinomial can fail if num_samples > num_elements or if probabilities sum to 0
                        # print(f"Warning: torch.multinomial failed for molecule {i}. Error: {e}")
                        # Fallback: simple random sampling from heteroatoms if weighted fails
                        if len(hetero_global_indices_in_mol) >= num_samples_from_hetero:
                            perm = torch.randperm(len(hetero_global_indices_in_mol), device=pos.device)
                            selected_hetero_indices = hetero_global_indices_in_mol[perm[:num_samples_from_hetero]]
                            selected_atoms_for_this_mol_list.append(selected_hetero_indices)
                            num_to_mask_for_this_mol -= len(selected_hetero_indices)

            # --- Step 2: Random sampling from non-heteroatoms if more atoms are needed ---
            if num_to_mask_for_this_mol > 0 and len(non_hetero_global_indices_in_mol) > 0:
                num_samples_from_non_hetero = min(num_to_mask_for_this_mol, len(non_hetero_global_indices_in_mol))

                perm_non_hetero = torch.randperm(len(non_hetero_global_indices_in_mol), device=pos.device)
                selected_non_hetero_indices = non_hetero_global_indices_in_mol[
                    perm_non_hetero[:num_samples_from_non_hetero]]
                selected_atoms_for_this_mol_list.append(selected_non_hetero_indices)
                num_to_mask_for_this_mol -= len(selected_non_hetero_indices)

            # --- Step 3: If still not enough (e.g. num_mask > total atoms in mol), take all available ---
            # This case should ideally be handled by ensuring num_mask is reasonable or by padding.
            # For simplicity, if we couldn't select self.num_mask atoms, we use what we have.

            num_selected_this_mol = len(
                torch.cat(selected_atoms_for_this_mol_list)) if selected_atoms_for_this_mol_list else 0

            if selected_atoms_for_this_mol_list:
                final_masked_indices_for_batch_list.append(torch.cat(selected_atoms_for_this_mol_list))

        # Combine selections from all molecules into a single flat tensor of global indices
        if not final_masked_indices_for_batch_list:
            #print("--- [DEBUG-M2] No atoms were sampled. Returning None. ---", flush=True)
            return None  # Or an empty dict to signify failure, handled by _ssp_gnn_pass

        current_iter_masked_indices_global = torch.cat(final_masked_indices_for_batch_list)
        # --- Prepare data for GNN input (unmasked part) ---
        all_atom_indices_global = torch.arange(pos.size(0), device=pos.device)
        is_unmasked_mask_bool = torch.ones(pos.size(0), dtype=torch.bool, device=pos.device)
        is_unmasked_mask_bool[current_iter_masked_indices_global] = False  # Mark selected atoms as masked
        unmasked_atom_indices_global = all_atom_indices_global[is_unmasked_mask_bool]

        ssp_input_pos = pos[unmasked_atom_indices_global]
        ssp_input_vel = vel[unmasked_atom_indices_global] if vel is not None else None
        ssp_input_batch_unmasked = batch_all_atoms[unmasked_atom_indices_global]
        ssp_input_node_atom_raw_z = node_atom_all_atoms_raw_z[unmasked_atom_indices_global]

        # --- Prepare target information for TransBlock ---
        ssp_target_prop_value = main_task_target_per_graph

        masked_atom_types_raw_z_flat = node_atom_all_atoms_raw_z[current_iter_masked_indices_global]
        batch_for_masked_atoms_flat = batch_all_atoms[current_iter_masked_indices_global]
        masked_atom_internal_indices_flat = self._map_atom_types_to_internal(masked_atom_types_raw_z_flat)
        masked_atom_type_embeddings_flat, _, _ = self.atom_embed_ssp_injection(masked_atom_internal_indices_flat)
        # 安全地获取批次中的图数量
        if main_task_target_per_graph is not None:
            # 对于QM9, target是存在的，形状为 (num_graphs, ...)，可以直接获取
            num_graphs_in_batch = main_task_target_per_graph.size(0)
        elif batch_all_atoms is not None and batch_all_atoms.numel() > 0:
            # 对于MD17, target是None, 我们从batch向量推断图的数量
            # .max() 返回最大索引，所以数量是 max_index + 1
            num_graphs_in_batch = batch_all_atoms.max().item() + 1
        else:
            # 边缘情况: 如果批次为空，则图数量为0
            num_graphs_in_batch = 0

        aggregated_masked_atom_info_per_graph = torch.zeros(
            num_graphs_in_batch, masked_atom_type_embeddings_flat.shape[-1],  # Use embedding dim
            device=pos.device, dtype=pos.dtype
        )  # Initialize with zeros

        if masked_atom_type_embeddings_flat.numel() > 0 and batch_for_masked_atoms_flat.numel() > 0:
            # scatter_reduce 'mean' requires counts, or use scatter with manual division for mean
            # For 'mean', scatter typically handles it correctly if counts are non-zero.
            # Ensure dim_size is correct.
            try:
                aggregated_masked_atom_info_per_graph = scatter(
                    source=masked_atom_type_embeddings_flat,
                    index=batch_for_masked_atoms_flat,
                    dim=0,
                    dim_size=num_graphs_in_batch,
                    reduce='mean'
                )
            except Exception as e:  # Catch potential scatter errors (e.g. index out of bounds if num_graphs_in_batch is wrong)
                # print(f"Error during scatter aggregation for masked atoms: {e}")
                # Keep aggregated_masked_atom_info_per_graph as zeros if scatter fails
                pass

        ssp_target_for_transblock = (ssp_target_prop_value, aggregated_masked_atom_info_per_graph)

        # --- Ground truth for position prediction head ---
        ssp_ground_truth_masked_positions = pos[current_iter_masked_indices_global]

        # print(f"[DEBUG] node_atom_all_atoms_raw_z: {node_atom_all_atoms_raw_z}")
        # print(f"[DEBUG] heteroatom_internal_indices: {self.heteroatom_internal_indices}")
        # print(f"[DEBUG] internal_atom_map_tensor: {self.internal_atom_map_tensor}")

        return {
            "pos_unmasked": ssp_input_pos,
            "batch_unmasked": ssp_input_batch_unmasked,
            "vel_unmasked": ssp_input_vel,
            "node_atom_unmasked_raw_z": ssp_input_node_atom_raw_z,
            "target_for_transblock": ssp_target_for_transblock,
            "gt_masked_positions": ssp_ground_truth_masked_positions,
            "batch_for_masked_gt": batch_for_masked_atoms_flat,
        }

    def _ssp_gnn_pass(self, ssp_data_dict: dict) -> torch.Tensor:
        """
        (V9 - 最终加固版)
        执行完整的EMPP(SSP)前向传播和损失计算。
        此版本使用拼接图保证邻居搜索的鲁棒性，并包含完整的调试和异常处理。
        """
        # 从字典中获取设备信息，以便在出错时返回正确设备上的张量
        device = ssp_data_dict["pos_unmasked"].device

        # --- 1. 对未掩码的图运行GNN骨干网络 ---
        try:
            unmasked_node_features, _ = self._run_gnn_backbone(
                pos=ssp_data_dict["pos_unmasked"],
                vel=ssp_data_dict["vel_unmasked"],
                batch=ssp_data_dict["batch_unmasked"],
                node_atom_raw_z=ssp_data_dict["node_atom_unmasked_raw_z"],
                ptr=None,  # ptr 在这里不是必需的，因为有 batch
                ssp_injection_target=ssp_data_dict["target_for_transblock"],
                debug_name="SSP_Task"
            )
        except Exception as e:
            print(f"!!! FATAL ERROR inside _run_gnn_backbone during SSP pass: {e} !!!", flush=True)
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, device=device)

        # --- 2. 检查GNN输出的数值稳定性 ---
        if torch.isnan(unmasked_node_features).any() or torch.isinf(unmasked_node_features).any():
            print("\n" + "=" * 50, flush=True)
            print("!!! FATAL DEBUG: NaN or Inf DETECTED in `unmasked_node_features` from GNN! !!!", flush=True)
            print("!!! The GNN backbone (TransBlocks) is numerically unstable on the masked graph. !!!", flush=True)
            print("=" * 50 + "\n", flush=True)
            # 返回0损失，防止程序崩溃
            return torch.tensor(0.0, device=device)

        # --- 3. 高效且可靠地查找邻居对 ---
        pos_unmasked = ssp_data_dict["pos_unmasked"]
        gt_masked_positions = ssp_data_dict["gt_masked_positions"]
        batch_unmasked = ssp_data_dict["batch_unmasked"]
        batch_for_masked_gt = ssp_data_dict["batch_for_masked_gt"]

        # a. 构建临时的“超级图”
        num_unmasked = pos_unmasked.shape[0]
        combined_pos = torch.cat([pos_unmasked, gt_masked_positions], dim=0)
        combined_batch = torch.cat([batch_unmasked, batch_for_masked_gt], dim=0)

        # b. 在“超级图”上运行 radius_graph
        edge_index_combined = pyg_nn.radius_graph(
            combined_pos, r=self.max_radius, batch=combined_batch, loop=False
        )

        # c. 从所有边中筛选出“跨界”边
        row, col = edge_index_combined
        mask_unmasked_to_masked = (row < num_unmasked) & (col >= num_unmasked)

        idx_unmasked_neighbors = row[mask_unmasked_to_masked]
        idx_masked_atoms = col[mask_unmasked_to_masked] - num_unmasked

        # 如果没有找到任何邻居对，打印警告并返回0损失
        if idx_unmasked_neighbors.numel() == 0:
            print("--- SSP WARNING: Found 0 neighbors in this batch! Returning 0 loss. ---", flush=True)
            return torch.tensor(0.0, device=device)

        # --- 4. 准备 pos_prediction_module 的输入 ---
        neighbor_features = unmasked_node_features[idx_unmasked_neighbors]
        neighbor_pos = pos_unmasked[idx_unmasked_neighbors]
        masked_pos_for_pairs = gt_masked_positions[idx_masked_atoms]
        gt_relative_pos = masked_pos_for_pairs - neighbor_pos

        # --- 5. 安全地调用预测模块计算损失 ---
        try:
            ssp_loss = self.pos_prediction_module(
                features_of_neighbors=neighbor_features,
                gt_relative_positions=gt_relative_pos
            )

            # 再次检查损失值是否有效
            if torch.isnan(ssp_loss).any() or torch.isinf(ssp_loss).any():
                print("!!! SSP WARNING: pos_prediction_module returned NaN/Inf loss. Setting to 0. !!!", flush=True)
                return torch.tensor(0.0, device=device)

            return ssp_loss

        except Exception as e:
            # 如果 pos_prediction 内部崩溃，打印详细错误并返回0损失
            print(f"!!! FATAL ERROR inside pos_prediction_module: {e} !!!", flush=True)
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, device=device)


    def _run_gnn_backbone(self, pos,vel, batch, node_atom_raw_z, ptr, ssp_injection_target=None,debug_name=""):
        #print(f"--- [DEBUG-GNN] Entering _run_gnn_backbone for: {debug_name} ---", flush=True)
        # 1. 初始原子嵌入
        node_atom_internal_idx = self._map_atom_types_to_internal(node_atom_raw_z)
        atom_embedding, atom_attr, _ = self.atom_embed_main(node_atom_internal_idx)
        check_for_nan(atom_embedding, "atom_embedding")

        # 2. 构建边和边属性
        edge_src, edge_dst = radius_graph(pos, r=self.max_radius, batch=batch, max_num_neighbors=1000)
        edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)

        # 归一化前检查零向量
        edge_length = edge_vec.norm(dim=1)
        edge_sh = o3.spherical_harmonics(
            l=self.irreps_edge_attr, x=edge_vec / (edge_length.unsqueeze(-1) + 1e-8),  # 安全归一化
            normalize=True, normalization='component'
        )
        edge_length_embedding = self.rbf(edge_length)
        check_for_nan(edge_sh, "edge_sh")
        check_for_nan(edge_length_embedding, "edge_length_embedding")

        # 3. 计算初始节点特征
        edge_deg_embed = self.edge_deg_embed(
            atom_embedding, edge_sh, edge_length_embedding, edge_src, edge_dst, batch
        )
        check_for_nan(edge_deg_embed, "edge_deg_embed")

        node_features = atom_embedding + edge_deg_embed
        check_for_nan(node_features, "initial node_features")

        # 4. 注入速度信息
        if self.task_type == 'md17' and vel is not None:
            vel_embedded = self.vel_embed(vel)
            node_features[:, self.l1_slice] += vel_embedded
            check_for_nan(node_features, "node_features after vel injection")

        # 5. 通过 TransBlock 堆栈
        node_attr_for_blocks = torch.ones_like(node_features.narrow(1, 0, 1))

        for i, blk in enumerate(self.blocks):
            node_features = blk(
                node_input=node_features,
                node_attr=node_attr_for_blocks,
                edge_src=edge_src, edge_dst=edge_dst, edge_attr=edge_sh,
                edge_scalars=edge_length_embedding, batch=batch,
                target=ssp_injection_target
            )
            check_for_nan(node_features, f"node_features after block {i}")

        # 6. Final Projection to Invariant Features
        node_inv_features = self.fc_inv_after_blocks(node_features)
        node_inv_features = self.norm_after_blocks(node_inv_features, batch=batch)
        node_inv_features = self.out_dropout_layer(node_inv_features)

        if torch.isnan(node_features).any():
            print(f"!!! FATAL (GNN): NaN detected in node_features for {debug_name} !!!", flush=True)
        return node_features, node_inv_features

    def get_graph_features_for_contrastive(self, pos: torch.Tensor,
                                           batch: torch.Tensor,
                                           node_atom_raw_z: torch.Tensor,
                                           ptr: Optional[torch.Tensor] = None,
                                           vel: Optional[torch.Tensor] = None  # <-- 为了统一，也接收vel
                                           ) -> torch.Tensor:
        """
        Extracts graph-level features suitable for contrastive learning.
        """
        if not self.enable_contrastive_learning:  # Check if contrastive learning is enabled for this model instance
            # It's better to handle this in engine.py by not calling this method if not enabled.
            # If called erroreously, raise an error or return a dummy tensor.
            # For now, let's assume engine.py handles the conditional call.
            pass

        # 1. Get base node features from GNN backbone using the existing method
        _, base_node_inv_features = self._run_gnn_backbone(
            pos=pos,
            vel=vel,  # 传递vel，即使CL任务本身不直接用，但保持GNN输入一致性
            batch=batch,
            node_atom_raw_z=node_atom_raw_z,
            ptr=ptr,
            ssp_injection_target=None
        )
        # base_node_inv_features shape: (num_nodes_in_batch, self.irreps_feature_before_head.dim)

        # 2. Graph Pooling
        # Ensure self.graph_pool_contrastive is initialized in __init__ if enable_contrastive_learning
        if not hasattr(self, 'graph_pool_contrastive'):
            raise AttributeError(
                "graph_pool_contrastive not initialized. Call this method only if enable_contrastive_learning was True during GAT init.")
        graph_level_features = self.graph_pool_contrastive(base_node_inv_features, batch)
        # graph_level_features shape: (batch_size, self.irreps_feature_before_head.dim)

        # 3. Projection Head for Contrastive Learning
        # Ensure self.projection_head_contrastive is initialized
        if not hasattr(self, 'projection_head_contrastive'):
            raise AttributeError("projection_head_contrastive not initialized.")
        projected_features_raw = self.projection_head_contrastive(graph_level_features)
        # projected_features_raw shape: (batch_size, self.contrastive_projection_dim)

        # 4. L2 Normalization
        projected_features_normalized = F.normalize(projected_features_raw, p=2, dim=1)

        return projected_features_normalized

    def forward(self,
                # 原始图数据
                pos, batch, node_atom, ptr,
                # MD17 任务需要速度
                vel=None,
                # 主任务目标，用于EMPP注入
                main_task_target_for_loss_and_ssp_injection=None,
                # 任务开关
                ssp_active_flag=False,
                contrastive_active_flag=False,
                # 对比学习的增强图数据，可选
                contrastive_augmented_data: Optional[tuple] = None
                ):
        outputs = {}
        # --- 1. 对原始图进行一次GNN Backbone计算 ---
        # ★ 核心优化: 一次调用，获取两种特征
        node_features_orig, node_inv_features_orig = self._run_gnn_backbone(
            pos, vel, batch, node_atom, ptr, ssp_injection_target=None
        )

        # --- 2. 计算主任务预测 ---
        if self.task_type == 'qm9':
            # QM9任务使用不变特征
            graph_features = self.scale_scatter_output(node_inv_features_orig, batch)
            outputs['main_pred'] = self.main_task_head(graph_features)
        elif self.task_type == 'md17':
            # ★ 关键修改: 只返回位移
            displacement = self.main_task_head(node_features_orig)
            outputs['main_pred'] = displacement  # <--- 现在 main_pred 是位移

        # --- 3. 计算对比学习任务 ---
        if contrastive_active_flag and self.training:
            # a) 原始视图的表征 z_i
            # ★ 优化点: 直接使用 run_gnn_backbone 返回的 node_inv_features_orig
            pooled_features_orig = self.graph_pool_contrastive(node_inv_features_orig, batch)
            projected_features_orig = self.projection_head_contrastive(pooled_features_orig)
            outputs['z_original_contrastive'] = F.normalize(projected_features_orig, p=2, dim=1)

            # b) 增强视图的表征 z_j
            if contrastive_augmented_data is not None:
                aug_pos, aug_batch, aug_z, aug_ptr = contrastive_augmented_data
                # ★ 优化点: 对增强图也一次性获取两种特征，这里只需要不变特征
                _, node_inv_features_aug = self._run_gnn_backbone(
                    aug_pos, vel, aug_batch, aug_z, aug_ptr, ssp_injection_target=None
                )
                pooled_features_aug = self.graph_pool_contrastive(node_inv_features_aug, aug_batch)
                projected_features_aug = self.projection_head_contrastive(pooled_features_aug)
                outputs['z_augmented_contrastive'] = F.normalize(projected_features_aug, p=2, dim=1)

        # --- 4. 计算EMPP辅助任务 ---
        if ssp_active_flag and self.training:
            ssp_data_dict = self.mask_position(
                pos, vel, batch, node_atom, ptr, main_task_target_for_loss_and_ssp_injection
            )
            if ssp_data_dict is not None:
                outputs['empp_loss'] = self._ssp_gnn_pass(ssp_data_dict)
                #print(f"--- [DEBUG-F3] _ssp_gnn_pass returned: {outputs['empp_loss']}", flush=True)
            else:
                #print("--- [DEBUG-F4] mask_position returned None. Setting empp_loss to 0. ---", flush=True)
                outputs['empp_loss'] = torch.tensor(0.0, device=pos.device)

        #print("--- [DEBUG-F5] Exiting forward(). ---", flush=True)
        return outputs


class pos_prediction(torch.nn.Module):
    def __init__(self, irreps_input_to_pos_pred, norm_for_pos_pred, base_atom_embedding_irreps,
                 res_s2grid=100, temperature_softmax=0.1, temperature_label_softmax=0.1,
                 radius_logit_range_max=6.0, num_radius_gaussians=128, radius_label_std=0.5,
                 num_s2_channels=32,
                 lmax_for_s2_basis=2):
        super().__init__()
        self.temperature = temperature_softmax
        self.temperature_label = temperature_label_softmax

        self.norm_layer_for_input_features = norm_for_pos_pred
        self.irreps_input_features = o3.Irreps(irreps_input_to_pos_pred)

        # --- Part 1: Gated MLP to process input features ---
        # Decompose input irreps for gating
        irreps_scalars_in, irreps_gates_in, irreps_gated_in = irreps2gate(self.irreps_input_features)

        # This is the irreps of the tensor *before* the gate activation
        irreps_pre_gate = (irreps_scalars_in + irreps_gates_in + irreps_gated_in).simplify()

        # This is the irreps of the tensor *after* the gate activation
        irreps_after_gate = (irreps_scalars_in + irreps_gated_in).simplify()

        self.fc_pre_gate = LinearRS(self.irreps_input_features, irreps_pre_gate, rescale=_RESCALE)
        self.gate_after_norm = Gate(
            irreps_scalars_in, [torch.nn.SiLU() for _, ir in irreps_scalars_in],
            irreps_gates_in, [torch.sigmoid for _, ir in irreps_gates_in],
            irreps_gated_in
        )

        # --- Part 2: Projection to final SH representation for S2 grid ---
        self.num_s2_channels = num_s2_channels
        self.lmax_for_s2_basis = lmax_for_s2_basis
        self.sh_basis_for_s2 = o3.Irreps.spherical_harmonics(lmax=self.lmax_for_s2_basis, p=1)

        # This is the target irreps for S2 grid projection
        self.irreps_internal_pos_pred = o3.Irreps(
            [(self.num_s2_channels, ir) for _, ir in self.sh_basis_for_s2]
        ).simplify()

        # ★★★ 关键修正 ★★★
        # The input to this layer is the output of the gate, which has `irreps_after_gate`
        self.fc_to_internal_irreps = LinearRS(
            irreps_in=irreps_after_gate,
            irreps_out=self.irreps_internal_pos_pred,
            rescale=_RESCALE
        )
        # ★★★★★★★★★★★★★★

        # --- Part 3: Heads for Radius and Direction prediction ---
        self.s2_grid_resolution = res_s2grid
        self.s2_transform = ToS2Grid_block(lmax=self.sh_basis_for_s2.lmax, res_beta=self.s2_grid_resolution)

        self.fc_s2_logits = nn.Sequential(
            nn.Linear(self.num_s2_channels, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )

        self.num_radius_bins = num_radius_gaussians
        scalar_channels_for_radius = self.num_s2_channels * self.sh_basis_for_s2.count('0e')
        self.fc_radius_logits = nn.Sequential(
            nn.Linear(scalar_channels_for_radius if scalar_channels_for_radius > 0 else 1, 64),
            nn.SiLU(),
            nn.Linear(64, self.num_radius_bins)
        )
        self.radius_bins_centers = torch.linspace(0.0, radius_logit_range_max, self.num_radius_bins)
        self.radius_label_std = radius_label_std
        self.kl_div_loss_fn = KLDivloss()

    def forward(self, features_of_neighbors: torch.Tensor, gt_relative_positions: torch.Tensor):

        if torch.isnan(features_of_neighbors).any() or torch.isinf(features_of_neighbors).any():
            print("!!! DEBUG (pos_pred): NaN/Inf detected in INPUT `features_of_neighbors`.")
            return torch.tensor(0.0, device=features_of_neighbors.device)
        # ----------------- 1. PREDICT Distributions from Neighbor Features -----------------

        # Process the input neighbor features through normalization and gating
        # print(f"--- DEBUG (pos_pred): Input features_of_neighbors norm = {features_of_neighbors.norm().item()}",
        #       flush=True)

        normalized_neighbor_features = self.norm_layer_for_input_features(features_of_neighbors)
        # print(f"--- DEBUG (pos_pred): After Norm, features norm = {normalized_neighbor_features.norm().item()}",
        #       flush=True)

        gated_features = self.gate_after_norm(self.fc_pre_gate(normalized_neighbor_features))
        #print(f"--- DEBUG (pos_pred): After Gate, features norm = {gated_features.norm().item()}", flush=True)

        processed_features_flat = self.fc_to_internal_irreps(gated_features)
        # print(
        #     f"--- DEBUG (pos_pred): After fc_to_internal_irreps, features norm = {processed_features_flat.norm().item()}",
        #     flush=True)

        if torch.isnan(processed_features_flat).any() or torch.isinf(processed_features_flat).any():
            print("!!! WARNING: NaN or Inf detected in features BEFORE radius/direction MLPs.")
            # 返回一个0损失，避免崩溃，让训练继续
            return torch.tensor(0.0, device=features_of_neighbors.device)
        # 使用 torch.clamp 来限制特征值的范围，防止它们过大
        # 这个范围需要实验，-10到10是一个比较安全的选择
        processed_features_flat = torch.clamp(processed_features_flat, -10, 10)

        # --- a) Predict Radius Distribution ---
        # Extract scalar part from processed features to predict radius
        scalar_channels_for_radius = self.num_s2_channels * self.sh_basis_for_s2.count('0e')
        scalar_features_for_radius = processed_features_flat.narrow(1, 0, scalar_channels_for_radius)
        radius_logits = self.fc_radius_logits(scalar_features_for_radius)

        if torch.isnan(radius_logits).any() or torch.isinf(radius_logits).any():
            print("!!! DEBUG (pos_pred): NaN/Inf detected in `radius_logits` BEFORE log_softmax.")
            # 打印出导致问题的输入特征，帮助分析
            problematic_indices = torch.where(torch.isnan(radius_logits) | torch.isinf(radius_logits))[0]
            print("Problematic input features (scalar_features_for_radius):")
            print(scalar_features_for_radius[problematic_indices])
            return torch.tensor(0.0, device=features_of_neighbors.device)

        log_softmax_radius_predicted = F.log_softmax(radius_logits, dim=-1)

        # --- b) Predict Direction Distribution ---
        # Reshape features to (Pairs, Channels, SH_dim) for S2 transformation
        num_pairs = processed_features_flat.shape[0]
        x_for_s2grid = processed_features_flat.reshape(num_pairs, self.num_s2_channels, self.sh_basis_for_s2.dim)

        # Project features onto the spherical grid
        s2_grid_output = self.s2_transform.ToGrid(x_for_s2grid)

        # Reshape and pass through MLP to get logits for each grid point
        s2_grid_output_permuted = s2_grid_output.permute(0, 2, 3, 1).contiguous()
        s2_grid_flattened_for_mlp = s2_grid_output_permuted.reshape(-1, self.num_s2_channels)
        direction_grid_logits = self.fc_s2_logits(s2_grid_flattened_for_mlp)

        # Reshape to (Pairs, NumGridPoints) and compute log softmax
        direction_logits_flat = direction_grid_logits.reshape(num_pairs, -1)
        log_softmax_direction_predicted = F.log_softmax(direction_logits_flat / self.temperature, dim=1)

        # ----------------- 2. COMPUTE Ground Truth Distributions -----------------

        # --- a) Safety Check for Zero Vectors ---
        # Detach from computation graph as it's a label
        label_relative_pos_vec = gt_relative_positions.detach()
        label_dist_norm_flat = label_relative_pos_vec.norm(dim=1)

        # Create a mask to filter out pairs where the relative position is a zero vector
        nonzero_mask = label_dist_norm_flat > 1e-6

        # If all pairs result in zero vectors (highly unlikely), return 0 loss to avoid crashing.
        if not torch.any(nonzero_mask):
            return torch.tensor(0.0, device=features_of_neighbors.device, dtype=torch.float)

        # --- b) Filter all relevant tensors using the mask ---
        filtered_log_softmax_direction = log_softmax_direction_predicted[nonzero_mask]
        filtered_log_softmax_radius = log_softmax_radius_predicted[nonzero_mask]

        filtered_relative_pos_vec = label_relative_pos_vec[nonzero_mask]
        filtered_dist_norm = label_dist_norm_flat[nonzero_mask].unsqueeze(-1)  # Shape: [N_filtered, 1]

        # --- c) Compute Target Distributions for the filtered data ---

        # Direction Target Distribution
        label_unit_direction_vec = filtered_relative_pos_vec / filtered_dist_norm
        label_sh_coeffs = o3.spherical_harmonics(
            l=self.sh_basis_for_s2,
            x=label_unit_direction_vec,
            normalize=True, normalization='component'
        )
        label_direction_on_grid = self.s2_transform.ToGrid(label_sh_coeffs.unsqueeze(1)).squeeze(1)
        label_direction_flat = label_direction_on_grid.reshape(label_direction_on_grid.shape[0], -1)
        label_direction_target_dist = F.softmax(label_direction_flat / self.temperature_label, dim=-1)

        # Radius Target Distribution (using Gaussian smearing)
        current_radius_bin_centers = self.radius_bins_centers.to(filtered_dist_norm.device)
        dist_from_centers = current_radius_bin_centers.view(1, -1) - filtered_dist_norm
        radius_gaussians_at_bins = torch.exp(-0.5 * (dist_from_centers / self.radius_label_std) ** 2)
        label_radius_target_dist = F.softmax(radius_gaussians_at_bins, dim=-1)

        # ----------------- 3. COMPUTE Final Loss -----------------

        # Calculate KL Divergence loss only on the valid (non-zero) pairs
        loss_direction = self.kl_div_loss_fn(filtered_log_softmax_direction, label_direction_target_dist)
        loss_radius = self.kl_div_loss_fn(filtered_log_softmax_radius, label_radius_target_dist)

        # Return the mean of the combined losses
        return (loss_direction.mean() + loss_radius.mean()) / 2.0

class reshape(nn.Module):
    def __init__(self, irreps):
        super().__init__()
        self.irreps = irreps
    
    def forward(self, x):

        ix = 0
        out = torch.tensor([], dtype=x.dtype, device=x.device)
        for mul, ir in self.irreps:
            d = ir.dim
            field = x[:, ix: ix + mul * d]
            ix = ix + mul * d

            field = field.reshape(-1, mul, d)
            out = torch.cat([out, field], dim=-1)
        return out

class KLDivloss(nn.Module):
    def __init__(self, epsilon=1e-10):  # epsilon not used in this KLDivLoss version
        super(KLDivloss, self).__init__()
        # reduction='batchmean' averages over batch and sums over features (last dim of input)
        # reduction='none' gives per-element loss. We need to sum over the distribution (last dim) then mean over batch.
        self.kl_loss_fn = torch.nn.KLDivLoss(reduction="none")

    def forward(self, log_probs_pred, target_probs, weights=None):
        # log_probs_pred: (Batch, NumClasses), target_probs: (Batch, NumClasses)
        # KLDiv(P || Q) = sum P_i * (log P_i - log Q_i)
        # PyTorch KLDivLoss(x, y) calculates sum y_i * (log y_i - x_i) where x is log_pred, y is target_true
        # So, it's effectively sum target_true * (log target_true - log_pred)
        # This is correct for D_KL(target_true || pred_from_log_pred)

        # Sum over the distribution dimension (NumClasses, e.g., NumGridPoints or NumRadiusBins)
        kl_div_per_sample = self.kl_loss_fn(log_probs_pred, target_probs).sum(dim=-1)

        if weights is not None:
            kl_div_per_sample = kl_div_per_sample * weights  # Element-wise if weights are per-sample

        return kl_div_per_sample  # Return per-sample loss, will be meaned outside or in the final return of pos_pred

class average(nn.Module):
    def __init__(self, irreps):
        super().__init__()
        self.irreps = irreps
    
    def forward(self, x):

        ix = 0
        out = torch.tensor([], device=x.device)
        for mul, ir in self.irreps:
            d = ir.dim
            field = x[:, ix: ix + mul * d]
            ix = ix + mul * d

            field = field.reshape(-1, mul, d)
            out = torch.cat([out, field.mean(1)], dim=-1)
        return out




@register_model
def graph_attention_transformer_nonlinear_l2(
        irreps_in, radius, num_basis=128,
        atomref=None, task_mean=None, task_std=None, ssp=None,
        # 接收来自 main.py args 的参数
        empp_num_mask=1,
        empp_ssp_feature_dim='16x0e',
        empp_atom_type_embed_irreps='16x0e',
        empp_prioritize_heteroatoms=True,  # 这个会影响 GAT 内部 heteroatom_internal_indices 的使用
        empp_pos_pred_num_s2_channels=32,
        empp_pos_pred_res_s2grid=100,
        empp_pos_pred_temp_softmax=0.1,
        empp_pos_pred_temp_label=0.1,
        enable_contrastive_learning=FALSE,
        contrastive_projection_dim=128, # New contrastive args
        drop_path=0.0,
        **kwargs):  # **kwargs 用于捕获其他未显式声明的参数

    # 固定的或从其他地方加载的配置
    # 注意：这些硬编码的映射和索引列表应该与你的数据和模型内部逻辑一致
    atom_map = {1: 0, 6: 1, 7: 2, 8: 3}

    # 步骤3：杂原子优先索引
    heteroatom_zs = [8, 7]  # O, N
    heteroatom_indices = [atom_map[z] for z in heteroatom_zs if z in atom_map]

    model = GraphAttentionTransformer(
        irreps_in=irreps_in,  # 通常是初始原子嵌入的Irreps
        irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,  # 模型的主体配置
        irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
        max_radius=radius,
        number_of_basis=num_basis, fc_neurons=[64, 64],
        irreps_feature='512x0e',
        irreps_head='32x0e+16x1e+8x2e', num_heads=4,
        rescale_degree=False, nonlinear_message=True,
        irreps_mlp_mid='384x0e+192x1e+96x2e',
        norm_layer='layer',
        alpha_drop=0.1, proj_drop=0.1, out_drop=0.1,
        drop_path_rate=0.1,
        mean=task_mean, std=task_std, scale=None, atomref=atomref,
        ssp=ssp,
        num_mask=empp_num_mask,  # 传递 num_mask
        # 传递掩码策略相关的固定配置 (或根据 empp_prioritize_heteroatoms 决定)
        qm9_atom_type_map=atom_map,
        heteroatom_mapped_indices=heteroatom_indices if empp_prioritize_heteroatoms else [],
        # 传递 TransBlock SSP 注入相关的配置
        ssp_feature_dim_transblock=empp_ssp_feature_dim,
        atom_type_embedding_irreps_transblock=empp_atom_type_embed_irreps,
        # 传递 pos_prediction 相关的配置
        ssp_pos_pred_num_s2_channels_config=empp_pos_pred_num_s2_channels,
        ssp_pos_pred_res_s2grid_config=empp_pos_pred_res_s2grid,
        ssp_pos_pred_temp_softmax_config=empp_pos_pred_temp_softmax,
        ssp_pos_pred_temp_label_config=empp_pos_pred_temp_label,
        enable_contrastive_learning=enable_contrastive_learning,  # Pass contrastive flag
        contrastive_projection_dim=contrastive_projection_dim,  # Pass projection dim
        **kwargs  # 将其他未处理的kwargs传递下去
    )
    return model




# Equiformer, L_max = 2, Bessel radial basis, dropout = 0.2
@register_model
def graph_attention_transformer_nonlinear_bessel_l2(**kwargs):
    """
    模型工厂: 使用Bessel基，并将alpha_drop硬编码为0.2。
    所有其他参数（包括你的SSP和CL开关）都从kwargs动态传入。
    """

    # --- 核心修改在这里 ---
    # 1. 明确设置 basis_type 为 'bessel'
    kwargs['basis_type'] = 'bessel'

    # 2. 明确设置 alpha_drop 为 0.2
    #    使用 .setdefault() 可以在不覆盖命令行传入值的情况下设置默认值
    #    但这里我们希望强制使用0.2，所以直接赋值
    kwargs['alpha_drop'] = 0.2
    # ---------------------

    # 创建模型实例，**kwargs会将所有配置（包括我们刚修改的）传递过去
    model = GraphAttentionTransformer(**kwargs)
    return model


# Equiformer, L_max = 2, Bessel radial basis, dropout = 0.1
@register_model
def graph_attention_transformer_nonlinear_bessel_l2_drop01(**kwargs):
    """
    模型工厂: 使用Bessel基，并将alpha_drop硬编码为0.1。
    """

    # --- 核心修改在这里 ---
    kwargs['basis_type'] = 'bessel'
    kwargs['alpha_drop'] = 0.1
    # ---------------------

    model = GraphAttentionTransformer(**kwargs)
    return model


# Equiformer, L_max = 2, Bessel radial basis, dropout = 0.0
@register_model
def graph_attention_transformer_nonlinear_bessel_l2_drop00(**kwargs):
    """
    模型工厂: 使用Bessel基，并将alpha_drop硬编码为0.0。
    """

    # --- 核心修改在这里 ---
    kwargs['basis_type'] = 'bessel'
    kwargs['alpha_drop'] = 0.0
    # ---------------------

    model = GraphAttentionTransformer(**kwargs)
    return model

# @register_model
# def graph_attention_transformer_nonlinear_exp_l2_md17(**kwargs):
#     # --- 核心修改在这里 ---
#     kwargs['basis_type'] = 'exp'
#     model = GraphAttentionTransformer(
#         **kwargs)
#     return model