import os
import weakref
from collections import defaultdict

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


_TRANSFER_STREAMS = {}
_PENDING_TRANSFER_STREAMS = {}


def _env_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


def maybe_empty_cache() -> None:
    if torch.cuda.is_available() and _env_enabled("CHUNKFT_EMPTY_CACHE", "0"):
        torch.cuda.empty_cache()


def should_save_fp32_groups() -> bool:
    return _env_enabled("CHUNKFT_SAVE_FP32_GROUPS", "0")


def should_sync_cuda() -> bool:
    return _env_enabled("CHUNKFT_CUDA_SYNC", "0")


def should_pin_memory() -> bool:
    default_value = "1" if should_async_offload() else "0"
    return torch.cuda.is_available() and _env_enabled("CHUNKFT_PIN_MEMORY", default_value)


def should_async_offload() -> bool:
    return torch.cuda.is_available() and _env_enabled("CHUNKFT_ASYNC_OFFLOAD", "1")


def should_prefetch() -> bool:
    return should_async_offload() and _env_enabled("CHUNKFT_ENABLE_PREFETCH", "1")


def should_lazy_resume() -> bool:
    return _env_enabled("CHUNKFT_LAZY_RESUME", "1")


def monkey_patches_enabled() -> bool:
    return _env_enabled("CHUNKFT_ENABLE_MONKEY_PATCHES", "1")


def get_offload_min_bytes() -> int:
    raw_value = os.environ.get("CHUNKFT_OFFLOAD_MIN_BYTES", "262144")
    try:
        return max(0, int(raw_value))
    except ValueError:
        return 262144


def get_cold_state_min_bytes() -> int:
    raw_value = os.environ.get("CHUNKFT_COLD_STATE_MIN_BYTES", "1048576")
    try:
        return max(0, int(raw_value))
    except ValueError:
        return 1048576


def get_cold_state_dtype():
    raw_value = os.environ.get("CHUNKFT_COLD_STATE_DTYPE", "bf16").lower()
    if raw_value in {"", "0", "false", "off", "none"}:
        return None
    if raw_value == "bf16":
        return torch.bfloat16
    if raw_value == "fp16":
        return torch.float16
    raise ValueError("`CHUNKFT_COLD_STATE_DTYPE` must be one of: bf16, fp16, none.")


def should_offload_tensor(tensor: torch.Tensor) -> bool:
    if tensor.device.type == "cpu" or tensor.is_sparse:
        return True
    return tensor.numel() * tensor.element_size() >= get_offload_min_bytes()


def _normalize_device(device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _get_transfer_stream(device: torch.device):
    device = _normalize_device(device)
    if device.type != "cuda" or not should_async_offload():
        return None
    key = (device.type, device.index)
    if key not in _TRANSFER_STREAMS:
        _TRANSFER_STREAMS[key] = torch.cuda.Stream(device=device)
    return _TRANSFER_STREAMS[key]


def _cleanup_pending_transfer(tensor_id: int, tensor_ref) -> None:
    entry = _PENDING_TRANSFER_STREAMS.get(tensor_id)
    if entry is None:
        return
    current_ref, _ = entry
    if current_ref is tensor_ref:
        _PENDING_TRANSFER_STREAMS.pop(tensor_id, None)


def _set_pending_transfer_stream(tensor: torch.Tensor, stream) -> None:
    if stream is None:
        return
    try:
        tensor_id = id(tensor)
        tensor_ref = weakref.ref(
            tensor,
            lambda ref, tensor_id=tensor_id: _cleanup_pending_transfer(tensor_id, ref),
        )
    except TypeError:
        return
    _PENDING_TRANSFER_STREAMS[tensor_id] = (tensor_ref, stream)


def _get_pending_transfer_stream(tensor: torch.Tensor, remove: bool = False):
    entry = _PENDING_TRANSFER_STREAMS.get(id(tensor))
    if entry is None:
        return None

    tensor_ref, stream = entry
    if tensor_ref() is not tensor:
        _PENDING_TRANSFER_STREAMS.pop(id(tensor), None)
        return None

    if remove:
        _PENDING_TRANSFER_STREAMS.pop(id(tensor), None)
    return stream


def _mark_pending_transfer(tensor: torch.Tensor, stream) -> torch.Tensor:
    _set_pending_transfer_stream(tensor, stream)
    return tensor


def wait_for_tensor_transfer(tensor: torch.Tensor, device=None) -> torch.Tensor:
    stream = _get_pending_transfer_stream(tensor, remove=True)
    if stream is None or not torch.cuda.is_available():
        return tensor
    wait_device = _normalize_device(device or (tensor.device if tensor.device.type == "cuda" else torch.device("cuda")))
    current_stream = torch.cuda.current_stream(device=wait_device)
    current_stream.wait_stream(stream)
    if tensor.device.type == "cuda":
        tensor.record_stream(current_stream)
    return tensor


def transfer_to_device(tensor: torch.Tensor, device: torch.device, prefetch: bool = False) -> torch.Tensor:
    device = _normalize_device(device)
    if tensor.device == device:
        if device.type == "cuda":
            wait_for_tensor_transfer(tensor, device=device)
        return tensor
    if tensor.device.type == "cpu":
        wait_for_tensor_transfer(tensor, device=device)
    non_blocking = tensor.device.type == "cpu" and getattr(tensor, "is_pinned", lambda: False)()
    if prefetch and non_blocking and device.type == "cuda":
        copy_stream = _get_transfer_stream(device)
        if copy_stream is not None:
            with torch.cuda.stream(copy_stream):
                gpu_tensor = tensor.to(device, non_blocking=True)
            return _mark_pending_transfer(gpu_tensor, copy_stream)
    return tensor.to(device, non_blocking=non_blocking)


def bucket_transfer_to_device(tensors, device: torch.device, prefetch: bool = False):
    device = _normalize_device(device)
    transferred_tensors = [None] * len(tensors)
    bucketed_indices = defaultdict(list)

    for index, tensor in enumerate(tensors):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.is_sparse
            or tensor.layout != torch.strided
            or tensor.device.type != "cpu"
        ):
            transferred_tensors[index] = transfer_to_device(tensor, device, prefetch=prefetch)
            continue
        bucketed_indices[(tensor.dtype, tensor.layout)].append(index)

    for indices in bucketed_indices.values():
        if len(indices) == 1:
            index = indices[0]
            transferred_tensors[index] = transfer_to_device(tensors[index], device, prefetch=prefetch)
            continue

        bucket = _flatten_dense_tensors([tensors[index] for index in indices])
        bucket = transfer_to_device(bucket, device, prefetch=prefetch)
        split_tensors = _unflatten_dense_tensors(bucket, [tensors[index] for index in indices])
        pending_stream = _get_pending_transfer_stream(bucket, remove=False)
        if pending_stream is not None:
            split_tensors = [_mark_pending_transfer(tensor, pending_stream) for tensor in split_tensors]
        for index, tensor in zip(indices, split_tensors):
            transferred_tensors[index] = tensor

    return transferred_tensors


def _can_reuse_cpu_buffer(buffer: torch.Tensor, tensor: torch.Tensor) -> bool:
    return (
        buffer is not None
        and buffer.device.type == "cpu"
        and buffer.shape == tensor.shape
        and buffer.dtype == tensor.dtype
        and buffer.layout == tensor.layout
    )


def _can_reuse_cpu_buffer_with_dtype(buffer: torch.Tensor, tensor: torch.Tensor, dtype: torch.dtype) -> bool:
    return (
        buffer is not None
        and buffer.device.type == "cpu"
        and buffer.shape == tensor.shape
        and buffer.dtype == dtype
        and buffer.layout == tensor.layout
    )


def transfer_to_cpu(tensor: torch.Tensor, buffer: torch.Tensor = None, async_transfer: bool = False) -> torch.Tensor:
    if tensor.device.type == "cpu":
        if should_pin_memory() and not tensor.is_sparse and hasattr(tensor, "pin_memory") and not tensor.is_pinned():
            return tensor.pin_memory()
        return tensor
    if tensor.is_sparse:
        return tensor.to("cpu", non_blocking=False)
    if not should_offload_tensor(tensor):
        return tensor

    pin_memory = should_pin_memory()
    if _can_reuse_cpu_buffer(buffer, tensor):
        reusable_buffer = buffer
        if pin_memory and hasattr(reusable_buffer, "is_pinned") and not reusable_buffer.is_pinned():
            reusable_buffer = torch.empty_like(tensor, device="cpu", pin_memory=True)
        if async_transfer and pin_memory and should_async_offload():
            copy_stream = _get_transfer_stream(tensor.device)
            if copy_stream is not None:
                with torch.cuda.stream(copy_stream):
                    reusable_buffer.copy_(tensor, non_blocking=True)
                return _mark_pending_transfer(reusable_buffer, copy_stream)
        reusable_buffer.copy_(tensor, non_blocking=pin_memory)
        return reusable_buffer

    if pin_memory:
        cpu_tensor = torch.empty_like(tensor, device="cpu", pin_memory=True)
        if async_transfer and should_async_offload():
            copy_stream = _get_transfer_stream(tensor.device)
            if copy_stream is not None:
                with torch.cuda.stream(copy_stream):
                    cpu_tensor.copy_(tensor, non_blocking=True)
                return _mark_pending_transfer(cpu_tensor, copy_stream)
        cpu_tensor.copy_(tensor, non_blocking=True)
        return cpu_tensor

    return tensor.to("cpu", non_blocking=False)


def offload_optimizer_state_to_cpu(tensor: torch.Tensor, buffer: torch.Tensor = None):
    original_dtype = tensor.dtype
    compression_dtype = get_cold_state_dtype()
    if (
        compression_dtype is None
        or tensor.device.type != "cuda"
        or tensor.is_sparse
        or not tensor.is_floating_point()
        or tensor.dtype != torch.float32
        or tensor.numel() * tensor.element_size() < get_cold_state_min_bytes()
    ):
        return transfer_to_cpu(tensor, buffer), original_dtype

    if _can_reuse_cpu_buffer_with_dtype(buffer, tensor, compression_dtype):
        cpu_tensor = buffer
    else:
        cpu_tensor = torch.empty_like(
            tensor,
            device="cpu",
            dtype=compression_dtype,
            pin_memory=should_pin_memory(),
        )
    cpu_tensor.copy_(tensor, non_blocking=False)
    return cpu_tensor, original_dtype


def restore_optimizer_state_to_device(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype = None):
    restored = transfer_to_device(tensor, device)
    if torch.device(device).type == "cuda":
        wait_for_tensor_transfer(restored, device=device)
    if dtype is not None and restored.dtype != dtype:
        restored = restored.to(dtype)
    return restored
