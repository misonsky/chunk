#coding=utf-8
import torch
import math
from torch import nn,Tensor, Size
import numbers
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn import init

import contextlib
import warnings
from typing import (
    Callable,
    ContextManager,
    List,
    Optional,
    Tuple,
    Union
)

from torch.utils.checkpoint import (
    check_backward_validity,
    _infer_device_type,
    _get_autocast_kwargs,
    _get_device_module,
    get_device_states,
    set_device_states,
    detach_variable,
    _supports_autocast,
    noop_context_fn,
    _DEFAULT_DETERMINISM_MODE,
    _checkpoint_without_reentrant_generator

)
_shape_t = Union[int, List[int], Size]


def _accumulate_new_grad(param: Optional[Tensor], grad: Optional[Tensor]) -> None:
    if param is None or grad is None:
        return

    existing_grad = getattr(param, "new_grad", None)
    if existing_grad is None:
        param.new_grad = grad
        return

    if grad.is_sparse:
        if not existing_grad.is_sparse:
            existing_grad.add_(grad.to_dense())
            return
        param.new_grad = (existing_grad.coalesce() + grad.coalesce()).coalesce()
        return

    if existing_grad.is_sparse:
        param.new_grad = existing_grad.to_dense()
        param.new_grad.add_(grad)
        return

    existing_grad.add_(grad)


def _is_full_range(upd_ran: Optional[Tuple[int, int]], size: int) -> bool:
    return upd_ran is None or len(upd_ran) == 0 or (upd_ran[0] == 0 and upd_ran[1] == size)


def _has_active_update_range(upd_ran: Optional[Tuple[int, int]], size: int) -> bool:
    if upd_ran is None or len(upd_ran) == 0:
        return size > 0
    return upd_ran[0] < upd_ran[1]


def _slice_last_dim(tensor: Tensor, upd_ran: Optional[Tuple[int, int]], size: Optional[int] = None) -> Tensor:
    if size is None:
        size = tensor.shape[-1]
    if _is_full_range(upd_ran, size):
        return tensor
    return tensor[..., upd_ran[0]:upd_ran[1]]

def _get_param_update_range(param: Optional[Tensor], size: int, param_name: str) -> Tuple[int, int]:
    if param is None:
        return (0, size)

    upd_ran = getattr(param, "upd_ran", None)
    if upd_ran is not None:
        if len(upd_ran) != 2:
            raise RuntimeError(f"Invalid `upd_ran` metadata on `{param_name}`: {upd_ran}.")
        return upd_ran

    # Frozen parameters can still participate in the backward graph because we
    # need their values to propagate input gradients. In that case we do not
    # need chunk metadata and can safely fall back to the full parameter range.
    if not getattr(param, "requires_grad", False):
        return (0, size)

    raise RuntimeError(
        f"Missing chunk metadata `upd_ran` on trainable parameter `{param_name}`. "
        "This usually means the parameter was not registered into ChunkFT before training started."
    )


class CustomEmbeddingGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input,weight,padding_idx,max_norm,norm_type,scale_grad_by_freq,sparse):
        ctx.indices = input
        ctx.weight = weight
        ctx.sparse = sparse
        ctx.scale_grad_by_freq = scale_grad_by_freq
        return F.embedding(
            input, weight, padding_idx, max_norm,norm_type, scale_grad_by_freq, sparse)
    @staticmethod
    def backward(ctx, grad_output):
        # print("======embedding=================this is auto backward=========================\n")
        indices=ctx.indices
        weight=ctx.weight
        sparse = ctx.sparse
        scale_grad_by_freq = ctx.scale_grad_by_freq
        embedding_matrix_size = weight.size()
        upd_ran = _get_param_update_range(weight, weight.shape[0], "embedding.weight")
        weight.grad = None
        if not _has_active_update_range(upd_ran, weight.shape[0]):
            return None,None,None,None,None,None,None
        start, end = upd_ran
        chunk_rows = end - start
        flat_indices = indices.reshape(-1)
        grad_rows = grad_output.reshape(-1, grad_output.shape[-1]).to(weight.dtype)
        selected_mask = (flat_indices >= start) & (flat_indices < end)
        if not torch.any(selected_mask):
            return None,None,None,None,None,None,None
        local_indices = (flat_indices[selected_mask] - start).to(torch.long)
        grad_rows = grad_rows[selected_mask]
        if sparse:
            if scale_grad_by_freq:
                _, inverse_indices, counts = local_indices.unique(return_inverse=True, return_counts=True)
                grad_rows = grad_rows / counts[inverse_indices].to(grad_rows.dtype).unsqueeze(1)
            weight_grad = torch.sparse_coo_tensor(
                local_indices.unsqueeze(0),
                grad_rows,
                (chunk_rows, embedding_matrix_size[1]),
                device=grad_output.device,
                dtype=weight.dtype,
            ).coalesce()
        else:
            weight_grad = torch.zeros(
                chunk_rows,
                embedding_matrix_size[1],
                device=grad_output.device,
                dtype=weight.dtype,
            )
            if scale_grad_by_freq:
                _, inverse_indices, counts = local_indices.unique(return_inverse=True, return_counts=True)
                freq = counts[inverse_indices].to(grad_rows.dtype).unsqueeze(1)
                weight_grad.index_add_(0, local_indices, grad_rows / freq)
            else:
                weight_grad.index_add_(0, local_indices, grad_rows)
        _accumulate_new_grad(weight, weight_grad)
        return None,None,None,None,None,None,None

class CustomGradFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx,input,weight,bias):
        """
            y=x * w^t + b
        """
        ctx.input=input
        ctx.weight=weight
        ctx.bias = bias
        return F.linear(input, weight, bias)
    @staticmethod
    def backward(ctx, grad_output):
        # print("=====layer==================this is auto backward=========================\n")
        # print("---------{}-----------------\n".format(torch.cuda.max_memory_allocated() /1024/1024/1024))
        if not getattr(ctx.weight, "requires_grad", False) and not (
            ctx.bias is not None and getattr(ctx.bias, "requires_grad", False)
        ):
            new_grad = None
            if ctx.input.requires_grad:
                if ctx.input.dim() == 3:
                    new_grad = torch.einsum("bsd,dt->bst", grad_output, ctx.weight)
                elif ctx.input.dim() == 2:
                    new_grad = torch.mm(grad_output, ctx.weight)
            return new_grad, None, None

        if getattr(ctx.weight, "strategy", 0):
            return CustomGradFunction._row_strategy(ctx,grad_output)
        else:
            return CustomGradFunction._column_strategy(ctx,grad_output)
    @staticmethod
    def _row_strategy(ctx,grad_output):
        input=ctx.input
        weight=ctx.weight
        bias=ctx.bias
        w_upd_ran = _get_param_update_range(weight, weight.shape[1], "linear.weight")
        if bias is not None:
            b_upd_ran = _get_param_update_range(bias, bias.shape[0], "linear.bias")
        weight.grad = None
        if bias is not None:
            bias.grad = None
        new_grad = None
        if input.dim()==3:
            if input.requires_grad:
                new_grad = torch.einsum("bsd,dt->bst",grad_output,weight)
            if weight.requires_grad and _has_active_update_range(w_upd_ran, weight.shape[1]):
                weight_grad = torch.einsum(
                    "bds,bst->dt",
                    grad_output.transpose(1,2),
                    _slice_last_dim(input, w_upd_ran, weight.shape[1]),
                )
                _accumulate_new_grad(weight, weight_grad)
            if bias is not None and bias.requires_grad and _has_active_update_range(b_upd_ran, bias.shape[0]):
                bias_grad = torch.sum(_slice_last_dim(grad_output, b_upd_ran, bias.shape[0]),dim=(0,1))
                _accumulate_new_grad(bias, bias_grad)
        elif input.dim()==2:
            if input.requires_grad:
                new_grad = torch.mm(grad_output,weight)
            if weight.requires_grad and _has_active_update_range(w_upd_ran, weight.shape[1]):
                weight_grad = torch.mm(grad_output.T,_slice_last_dim(input, w_upd_ran, weight.shape[1]))
                _accumulate_new_grad(weight, weight_grad)
            if bias is not None and bias.requires_grad and _has_active_update_range(b_upd_ran, bias.shape[0]):
                bias_grad = torch.sum(_slice_last_dim(grad_output, b_upd_ran, bias.shape[0]),dim=0)
                _accumulate_new_grad(bias, bias_grad)
        # else:
        #     print(input.size())
        #     print(weight.size())
        return new_grad, None, None
    @staticmethod
    def _column_strategy(ctx,grad_output):
        input=ctx.input
        weight=ctx.weight
        bias=ctx.bias
        w_upd_ran = _get_param_update_range(weight, weight.shape[0], "linear.weight")
        if bias is not None:
            b_upd_ran = _get_param_update_range(bias, bias.shape[0], "linear.bias")
        weight.grad = None
        if bias is not None:
            bias.grad = None
        new_grad = None
        if input.dim() == 3:
            if input.requires_grad:
                new_grad=torch.einsum("bsd,dt->bst",grad_output,weight)
            if weight.requires_grad and _has_active_update_range(w_upd_ran, weight.shape[0]):
                weight_grad = torch.einsum(
                    "bds,bst->dt",
                    _slice_last_dim(grad_output, w_upd_ran, weight.shape[0]).transpose(1,2),
                    input,
                )
                _accumulate_new_grad(weight, weight_grad)
            if bias is not None and bias.requires_grad and _has_active_update_range(b_upd_ran, bias.shape[0]):
                bias_grad = torch.sum(_slice_last_dim(grad_output, b_upd_ran, bias.shape[0]),dim=(0,1))
                _accumulate_new_grad(bias, bias_grad)
        if input.dim() == 2:
            if input.requires_grad:
                new_grad=torch.mm(grad_output,weight)
            if weight.requires_grad and _has_active_update_range(w_upd_ran, weight.shape[0]):
                weight_grad = torch.mm(_slice_last_dim(grad_output, w_upd_ran, weight.shape[0]).T,input)
                _accumulate_new_grad(weight, weight_grad)
            if bias is not None and bias.requires_grad and _has_active_update_range(b_upd_ran, bias.shape[0]):
                bias_grad = torch.sum(_slice_last_dim(grad_output, b_upd_ran, bias.shape[0]),dim=0)
                _accumulate_new_grad(bias, bias_grad)
        return  new_grad,None,None
class CustomLlamaRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states,weight,eps):
        ctx.input = hidden_states
        ctx.weight = weight
        ctx.eps = eps

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return weight * hidden_states.to(input_dtype)
    @staticmethod
    def backward(ctx, grad_output):
        # print("=======layerNormal================this is auto backward=========================\n")
        input = ctx.input
        weight = ctx.weight
        eps = ctx.eps
        
        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + eps)
        normalized_input = (input - mean) / std
        N = input.shape[-1]
        
        upd_ran = _get_param_update_range(ctx.weight, weight.shape[0], "llama_rms_norm.weight")
        
        weight.grad = None
        # Calculate grad_weight
        if weight is not None:
            grad_normalized_input = grad_output * weight
            if _has_active_update_range(upd_ran, weight.shape[0]) and input.dim()==3:
                grad_slice = _slice_last_dim(grad_output, upd_ran, weight.shape[0])
                normalized_slice = _slice_last_dim(normalized_input, upd_ran, weight.shape[0])
                weight_grad = torch.sum(grad_slice * normalized_slice, dim=(0,1))
                _accumulate_new_grad(weight, weight_grad)
            elif _has_active_update_range(upd_ran, weight.shape[0]) and input.dim()==2:
                grad_slice = _slice_last_dim(grad_output, upd_ran, weight.shape[0])
                normalized_slice = _slice_last_dim(normalized_input, upd_ran, weight.shape[0])
                weight_grad = torch.sum(grad_slice * normalized_slice, dim=0)
                _accumulate_new_grad(weight, weight_grad)
        else:
            grad_normalized_input = grad_output

        # Calculate grad_input
        grad_var = torch.sum(grad_normalized_input * (input - input.mean(dim=-1, keepdim=True)) * -0.5 * (std ** -3), dim=-1, keepdim=True)
        grad_mean = torch.sum(grad_normalized_input * -1 / std, dim=-1, keepdim=True) + grad_var * torch.mean(-2 * (input - input.mean(dim=-1, keepdim=True)), dim=-1, keepdim=True)
        grad_input = grad_normalized_input / std + grad_var * 2 * (input - input.mean(dim=-1, keepdim=True)) / N + grad_mean / N

        return grad_input, None, None

# class CustomLayerNormGrad(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, normalized_shape, weight, bias, eps):
#         ctx.input = input
#         ctx.weight = weight
#         ctx.bias = bias
#         ctx.eps = eps
#         return F.layer_norm(input, normalized_shape, weight, bias, eps)
#     @staticmethod
#     def backward(ctx, grad_output):
#         # print("======layerNormal=================this is auto backward=========================\n")
#         input = ctx.input
#         weight = ctx.weight
#         bias = ctx.bias
#         eps = ctx.eps

#         mean = input.mean(dim=-1, keepdim=True)
#         var = input.var(dim=-1, keepdim=True, unbiased=False)
#         std = torch.sqrt(var + eps)
#         normalized_input = (input - mean) / std
#         N = input.shape[-1]

#         upd_ran = _get_param_update_range(ctx.weight, weight.shape[0], "layer_norm.weight")

#         weight.grad = bias.grad = None
#         # Calculate grad_weight
#         if weight is not None:
#             grad_normalized_input = grad_output * weight
#             if _has_active_update_range(upd_ran, weight.shape[0]) and input.dim()==3:
#                 grad_slice = _slice_last_dim(grad_output, upd_ran, weight.shape[0])
#                 normalized_slice = _slice_last_dim(normalized_input, upd_ran, weight.shape[0])
#                 weight_grad = torch.sum(grad_slice * normalized_slice, dim=(0,1))
#                 _accumulate_new_grad(weight, weight_grad)
#             elif _has_active_update_range(upd_ran, weight.shape[0]) and input.dim()==2:
#                 grad_slice = _slice_last_dim(grad_output, upd_ran, weight.shape[0])
#                 normalized_slice = _slice_last_dim(normalized_input, upd_ran, weight.shape[0])
#                 weight_grad = torch.sum(grad_slice * normalized_slice, dim=0)
#                 _accumulate_new_grad(weight, weight_grad)
#         else:
#             grad_normalized_input = grad_output
#         # Calculate grad_bias
#         if bias is not None and _has_active_update_range(upd_ran, bias.shape[0]):
#             if input.dim()==3:
#                 bias_grad = torch.sum(_slice_last_dim(grad_output, upd_ran, bias.shape[0]), dim=(0,1))
#             elif input.dim()==2:
#                 bias_grad = torch.sum(_slice_last_dim(grad_output, upd_ran, bias.shape[0]), dim=0)
#             _accumulate_new_grad(bias, bias_grad)
#         # Calculate grad_input
#         grad_var = torch.sum(grad_normalized_input * (input - input.mean(dim=-1, keepdim=True)) * -0.5 * (std ** -3), dim=-1, keepdim=True)
#         grad_mean = torch.sum(grad_normalized_input * -1 / std, dim=-1, keepdim=True) + grad_var * torch.mean(-2 * (input - input.mean(dim=-1, keepdim=True)), dim=-1, keepdim=True)
#         grad_input = grad_normalized_input / std + grad_var * 2 * (input - input.mean(dim=-1, keepdim=True)) / N + grad_mean / N

#         return grad_input, None, None, None, None
class CustomLayerNormGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, weight, bias, eps):
        ctx.input = input
        ctx.weight = weight
        ctx.bias = bias
        ctx.eps = eps
        return F.layer_norm(input, normalized_shape, weight, bias, eps)
    @staticmethod
    def backward(ctx, grad_output):
        # print("======layerNormal=================this is auto backward=========================\n")
        input = ctx.input
        weight = ctx.weight
        bias = ctx.bias
        eps = ctx.eps

        input_dtype = input.dtype
        input_fp32 = input.to(torch.float32)
        grad_output_fp32 = grad_output.to(torch.float32)
        weight_fp32 = weight.to(torch.float32) if weight is not None else None

        mean = input_fp32.mean(dim=-1, keepdim=True)
        var = input_fp32.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + eps)
        normalized_input = (input_fp32 - mean) / std
        N = input.shape[-1]

        upd_ran = _get_param_update_range(ctx.weight, weight.shape[0], "layer_norm.weight")

        weight.grad = bias.grad = None
        # Calculate grad_weight
        if weight is not None:
            grad_normalized_input = grad_output_fp32 * weight_fp32
            if _has_active_update_range(upd_ran, weight.shape[0]) and input.dim()==3:
                grad_slice = _slice_last_dim(grad_output_fp32, upd_ran, weight.shape[0])
                normalized_slice = _slice_last_dim(normalized_input, upd_ran, weight.shape[0])
                weight_grad = torch.sum(grad_slice * normalized_slice, dim=(0,1)).to(weight.dtype)
                _accumulate_new_grad(weight, weight_grad)
            elif _has_active_update_range(upd_ran, weight.shape[0]) and input.dim()==2:
                grad_slice = _slice_last_dim(grad_output_fp32, upd_ran, weight.shape[0])
                normalized_slice = _slice_last_dim(normalized_input, upd_ran, weight.shape[0])
                weight_grad = torch.sum(grad_slice * normalized_slice, dim=0).to(weight.dtype)
                _accumulate_new_grad(weight, weight_grad)
        else:
            grad_normalized_input = grad_output_fp32
        # Calculate grad_bias
        if bias is not None and _has_active_update_range(upd_ran, bias.shape[0]):
            if input.dim()==3:
                bias_grad = torch.sum(_slice_last_dim(grad_output_fp32, upd_ran, bias.shape[0]), dim=(0,1)).to(bias.dtype)
            elif input.dim()==2:
                bias_grad = torch.sum(_slice_last_dim(grad_output_fp32, upd_ran, bias.shape[0]), dim=0).to(bias.dtype)
            _accumulate_new_grad(bias, bias_grad)
        # Calculate grad_input
        centered_input = input_fp32 - mean
        grad_var = torch.sum(grad_normalized_input * centered_input * -0.5 * (std ** -3), dim=-1, keepdim=True)
        grad_mean = torch.sum(grad_normalized_input * -1 / std, dim=-1, keepdim=True) + grad_var * torch.mean(-2 * centered_input, dim=-1, keepdim=True)
        grad_input = grad_normalized_input / std + grad_var * 2 * centered_input / N + grad_mean / N

        return grad_input.to(input_dtype), None, None, None, None
class Embedding(nn.Module):
    __constants__ = ['num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']
    num_embeddings: int
    embedding_dim: int
    padding_idx: Optional[int]
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    freeze: bool
    sparse: bool
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None, _freeze: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs),
                                    requires_grad=not _freeze)
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = Parameter(_weight, requires_grad=not _freeze)

        self.sparse = sparse

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        return CustomEmbeddingGrad.apply(input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)
class Linear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return CustomGradFunction.apply(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return CustomLlamaRMSNorm.apply(hidden_states,self.weight,self.variance_epsilon)

class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool
    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            if bias:
                self.bias = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)
    
    def forward(self, input: Tensor) -> Tensor:
        return CustomLayerNormGrad.apply(input, self.normalized_shape, self.weight, self.bias, self.eps)
    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
        ctx.device = _infer_device_type(*args)
        ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs(
            ctx.device
        )
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_device_in_fwd = False
            device_module = _get_device_module(ctx.device)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                ctx.fwd_devices, ctx.fwd_device_states = get_device_states(*args)

        # Save non-tensor inputs in ctx, keep a placeholder None for tensors
        # to be filled out during the backward.
        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        ctx.save_for_backward(*tensor_inputs)

        with torch.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument."
            )
        # Copy the list to avoid modifying original list.
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors
        device_module = _get_device_module(ctx.device)

        # Fill in inputs with appropriate saved tensors.
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]

        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_device_in_fwd:
            rng_devices = ctx.fwd_devices
        with torch.random.fork_rng(
            devices=rng_devices, enabled=ctx.preserve_rng_state, device_type=ctx.device
        ):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    set_device_states(ctx.fwd_devices, ctx.fwd_device_states)
            detached_inputs = detach_variable(tuple(inputs))

            device_autocast_ctx = device_module.amp.autocast(
                **ctx.device_autocast_kwargs
            ) if _supports_autocast(ctx.device) else contextlib.nullcontext()
            with torch.enable_grad(), device_autocast_ctx, \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # run backward() with only tensor that requires grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "none of output has requires_grad=True,"
                " this checkpoint() is not necessary"
            )
        # print("*****************this is checkpoint function*****************\n")
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None
            for inp in detached_inputs
        )

        return (None, None) + grads

@torch._disable_dynamo
def checkpoint(
    function,
    *args,
    use_reentrant: Optional[bool] = None,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    **kwargs
):
    
    if use_reentrant is None:
        warnings.warn(
            "torch.utils.checkpoint: please pass in use_reentrant=True or "
            "use_reentrant=False explicitly. The default value of use_reentrant "
            "will be updated to be False in the future. To maintain current "
            "behavior, pass use_reentrant=True. It is recommended that you use "
            "use_reentrant=False. Refer to docs for more details on the "
            "differences between the two variants."
        )
        use_reentrant = True
    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop("preserve_rng_state", True)
    if kwargs and use_reentrant:
        raise ValueError(
            "Unexpected keyword arguments: " + ",".join(arg for arg in kwargs)
        )

    if use_reentrant:
        if context_fn is not noop_context_fn or debug is not False:
            raise ValueError(
                "Passing `context_fn` or `debug` is only supported when "
                "use_reentrant=False."
            )
        return CheckpointFunction.apply(function, preserve, *args)
    else:
        gen = _checkpoint_without_reentrant_generator(
            function, preserve, context_fn, determinism_check, debug, *args, **kwargs
        )
        # Runs pre-forward logic
        next(gen)
        ret = function(*args, **kwargs)
        # Runs post-forward logic
        try:
            next(gen)
        except StopIteration:
            return ret
