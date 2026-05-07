#coding=utf-8
import functools
import re
from peft import (
    PeftType,
    LoraConfig,
    TaskType,
    get_peft_model,
    AdaLoraConfig,
    IA3Config,
    LoraConfig,
    PromptEncoderConfig,
    PrefixTuningConfig,
    PromptTuningConfig
)
from peft.tuners.adalora import RankAllocator
import transformers
import torch
from .layer import Embedding,Linear,LayerNorm,checkpoint

VALID_CHUNK_STRATEGIES = {"row", "column"}
VALID_GRADIENT_CHECKPOINT_MODES = {"tail", "head", "uniform"}

TRANSFORMER_LAYER_PATHS = {
    "bert": ("bert.encoder.layer", "encoder.layer"),
    "roberta": ("roberta.encoder.layer", "encoder.layer"),
    "gpt2": ("transformer.h", "h"),
    "gptneox": ("gpt_neox.layers", "layers"),
    "gptneo": ("transformer.h", "h"),
    "opt": ("model.decoder.layers", "decoder.layers"),
    "llamafamily": ("model.layers", "layers"),
}

def rebuild_layer():
    torch.nn.Embedding = Embedding
    torch.nn.Linear = Linear
    torch.nn.LayerNorm = LayerNorm


def _iter_tensors(value):
    if torch.is_tensor(value):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _resolve_attr_path(module, attr_path):
    current = module
    for attr in attr_path.split("."):
        if not hasattr(current, attr):
            return None
        current = getattr(current, attr)
    return current


def _parse_layer_spec(layer_spec, total_layers):
    indices = set()
    if layer_spec is None:
        return []

    for chunk in str(layer_spec).split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            start_str, end_str = item.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid gradient checkpoint layer range `{item}`.")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(item))

    invalid = sorted(index for index in indices if index < 0 or index >= total_layers)
    if invalid:
        raise ValueError(
            f"gradient_checkpointing_layers contains out-of-range indices {invalid} for {total_layers} layers."
        )
    return sorted(indices)


def _select_layer_indices(total_layers, ratio, mode):
    if ratio <= 0:
        return []
    selected_count = max(1, min(total_layers, int(round(total_layers * ratio))))
    if selected_count >= total_layers:
        return list(range(total_layers))
    if mode == "tail":
        return list(range(total_layers - selected_count, total_layers))
    if mode == "head":
        return list(range(selected_count))
    if mode == "uniform":
        if selected_count == 1:
            return [total_layers - 1]
        return sorted({round(i * (total_layers - 1) / (selected_count - 1)) for i in range(selected_count)})
    raise ValueError(f"Unsupported gradient checkpointing mode `{mode}`.")


def _normalize_checkpoint_model_type(config):
    model_type = getattr(config, "model_type", "")
    model_type = str(model_type).lower()
    if "gpt-neox" in model_type or "gpt_neox" in model_type:
        return "gptneox"
    if "gpt-neo" in model_type or "gpt_neo" in model_type:
        return "gptneo"
    if "llama" in model_type:
        return "llamafamily"
    return model_type


def _get_transformer_layers(model, config):
    model_type = _normalize_checkpoint_model_type(config)
    if model_type not in TRANSFORMER_LAYER_PATHS:
        return []

    candidate_roots = [model]
    if hasattr(model, "get_base_model"):
        base_model = model.get_base_model()
        if base_model is not model:
            candidate_roots.insert(0, base_model)

    for root in candidate_roots:
        for attr_path in TRANSFORMER_LAYER_PATHS[model_type]:
            resolved = _resolve_attr_path(root, attr_path)
            if resolved is not None:
                return list(resolved)
    return []


def _estimate_module_bytes(module):
    total_bytes = 0
    for parameter in module.parameters(recurse=False):
        total_bytes += parameter.numel() * parameter.element_size()
    return total_bytes


def _split_layer_indices_by_byte_budget(transformer_layers, selected_indices, chunk_num):
    if chunk_num <= 1 or not selected_indices:
        return {0: list(selected_indices)}

    layer_specs = []
    total_bytes = 0
    for layer_index in selected_indices:
        layer_bytes = max(1, _estimate_module_bytes(transformer_layers[layer_index]))
        layer_specs.append((layer_index, layer_bytes))
        total_bytes += layer_bytes

    if total_bytes <= 0:
        return {chunk_idx: [] for chunk_idx in range(chunk_num)}

    chunk_boundaries = [total_bytes * idx / chunk_num for idx in range(chunk_num + 1)]
    schedule = {chunk_idx: [] for chunk_idx in range(chunk_num)}
    allocated_bytes = 0.0
    current_chunk = 0
    for layer_index, layer_bytes in layer_specs:
        layer_midpoint = allocated_bytes + layer_bytes / 2.0
        while current_chunk + 1 < chunk_num and layer_midpoint >= chunk_boundaries[current_chunk + 1]:
            current_chunk += 1
        schedule[current_chunk].append(layer_index)
        allocated_bytes += layer_bytes
    return schedule


def _wrap_module_forward_with_checkpoint(module):
    if getattr(module, "_chunkft_original_forward", None) is not None:
        return

    original_forward = module.forward

    @functools.wraps(original_forward)
    def checkpointed_forward(*args, **kwargs):
        if not getattr(module, "_chunkft_checkpoint_enabled", True):
            return original_forward(*args, **kwargs)
        if not module.training or not torch.is_grad_enabled():
            return original_forward(*args, **kwargs)
        if not any(tensor.requires_grad for tensor in _iter_tensors((args, kwargs))):
            return original_forward(*args, **kwargs)
        return checkpoint(original_forward, *args, use_reentrant=False, **kwargs)

    module._chunkft_original_forward = original_forward
    module._chunkft_checkpoint_enabled = True
    module.forward = checkpointed_forward


def set_chunk_checkpoint_layers(model, chunk_idx=0):
    checkpoint_state = getattr(model, "_chunkft_selective_gradient_checkpointing", None)
    if checkpoint_state is None:
        return []

    wrapped_layers = checkpoint_state.get("wrapped_layers", {})
    if not wrapped_layers:
        return []

    if checkpoint_state.get("chunk_aware", False):
        schedule = checkpoint_state.get("schedule", {})
        active_indices = schedule.get(chunk_idx, [])
    else:
        active_indices = checkpoint_state.get("indices", [])

    active_index_set = set(active_indices)
    for layer_index, module in wrapped_layers.items():
        module._chunkft_checkpoint_enabled = layer_index in active_index_set

    checkpoint_state["active_indices"] = list(active_indices)
    checkpoint_state["active_chunk_idx"] = int(chunk_idx)
    return list(active_indices)


def apply_gradient_checkpointing_strategy(model, config, model_args, training_args, logger=None):
    if not getattr(training_args, "gradient_checkpointing", False):
        return False

    explicit_layers = getattr(model_args, "gradient_checkpointing_layers", None)
    ratio = getattr(model_args, "gradient_checkpointing_ratio", 1.0)
    mode = getattr(model_args, "gradient_checkpointing_mode", "tail")

    if explicit_layers in ("", None) and ratio >= 1.0:
        return False

    if mode not in VALID_GRADIENT_CHECKPOINT_MODES:
        valid_modes = ", ".join(sorted(VALID_GRADIENT_CHECKPOINT_MODES))
        raise ValueError(f"`gradient_checkpointing_mode` must be one of: {valid_modes}.")

    transformer_layers = _get_transformer_layers(model, config)
    if not transformer_layers:
        raise ValueError(
            f"Selective gradient checkpointing is not supported for model type `{getattr(config, 'model_type', 'unknown')}`."
        )

    if explicit_layers not in ("", None):
        selected_indices = _parse_layer_spec(explicit_layers, len(transformer_layers))
    else:
        selected_indices = _select_layer_indices(len(transformer_layers), ratio, mode)

    if not selected_indices:
        if logger is not None:
            logger.warning("Selective gradient checkpointing selected no layers; falling back to no checkpointing.")
        return False

    wrapped_layers = {}
    for layer_index in selected_indices:
        _wrap_module_forward_with_checkpoint(transformer_layers[layer_index])
        wrapped_layers[layer_index] = transformer_layers[layer_index]

    chunk_aware = bool(getattr(model_args, "chunk_tuning", False) and getattr(model_args, "chunk_num", 1) > 1)
    schedule = None
    active_indices = list(selected_indices)
    if chunk_aware:
        schedule = _split_layer_indices_by_byte_budget(transformer_layers, selected_indices, model_args.chunk_num)
        active_indices = schedule.get(0, [])

    if hasattr(model, "config") and getattr(model.config, "use_cache", None):
        model.config.use_cache = False
    if hasattr(config, "use_cache") and getattr(config, "use_cache", None):
        config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model._chunkft_selective_gradient_checkpointing = {
        "indices": selected_indices,
        "mode": mode,
        "ratio": ratio,
        "chunk_aware": chunk_aware,
        "schedule": schedule,
        "wrapped_layers": wrapped_layers,
        "active_indices": list(active_indices),
        "active_chunk_idx": 0,
    }
    set_chunk_checkpoint_layers(model, 0)
    if logger is not None:
        if chunk_aware:
            logger.info(
                "Enabled chunk-aware selective gradient checkpointing on %s/%s transformer layers. Initial active layers for chunk 0: %s",
                len(selected_indices),
                len(transformer_layers),
                active_indices,
            )
        else:
            logger.info(
                "Enabled selective gradient checkpointing on %s/%s transformer layers: %s",
                len(selected_indices),
                len(transformer_layers),
                selected_indices,
            )
    return True

def normalize_chunk_args(model_args, logger=None):
    group_element = getattr(model_args, "group_element", None)
    if group_element is not None:
        if model_args.chunk_num != 1 and model_args.chunk_num != group_element:
            if logger is not None:
                logger.warning(
                    "Both `--group_element=%s` and `--chunk_num=%s` were provided; keeping `--chunk_num`.",
                    group_element,
                    model_args.chunk_num,
                )
        else:
            model_args.chunk_num = group_element
            if logger is not None:
                logger.warning("`--group_element` is deprecated; please use `--chunk_num`.")

    optimizer_strategy = getattr(model_args, "optimizer_strategy", None)
    if optimizer_strategy is not None:
        if model_args.chunk_strategy != "row" and model_args.chunk_strategy != optimizer_strategy:
            if logger is not None:
                logger.warning(
                    "Both `--optimizer_strategy=%s` and `--chunk_strategy=%s` were provided; keeping `--chunk_strategy`.",
                    optimizer_strategy,
                    model_args.chunk_strategy,
                )
        else:
            model_args.chunk_strategy = optimizer_strategy
            if logger is not None:
                logger.warning("`--optimizer_strategy` is deprecated; please use `--chunk_strategy`.")

    if model_args.chunk_num < 1:
        raise ValueError("`chunk_num` must be a positive integer.")
    if getattr(model_args, "chunk_update_interval", 1) < 1:
        raise ValueError("`chunk_update_interval` must be a positive integer.")
    if model_args.chunk_strategy not in VALID_CHUNK_STRATEGIES:
        valid = ", ".join(sorted(VALID_CHUNK_STRATEGIES))
        raise ValueError(f"`chunk_strategy` must be one of: {valid}.")
    if hasattr(model_args, "gradient_checkpointing_ratio"):
        ratio = model_args.gradient_checkpointing_ratio
        if ratio < 0 or ratio > 1:
            raise ValueError("`gradient_checkpointing_ratio` must be between 0 and 1.")
    if hasattr(model_args, "gradient_checkpointing_mode"):
        mode = model_args.gradient_checkpointing_mode
        if mode not in VALID_GRADIENT_CHECKPOINT_MODES:
            valid_modes = ", ".join(sorted(VALID_GRADIENT_CHECKPOINT_MODES))
            raise ValueError(f"`gradient_checkpointing_mode` must be one of: {valid_modes}.")

    return model_args
def _update_ipt(self, model):
    # Update the sensitivity and uncertainty for every weight
    for n, p in model.named_parameters():
        if "lora_" in n and self.adapter_name in n:
            if n not in self.ipt:
                self.ipt[n] = torch.zeros_like(p)
                self.exp_avg_ipt[n] = torch.zeros_like(p)
                self.exp_avg_unc[n] = torch.zeros_like(p)
            with torch.no_grad():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                self.ipt[n] = (p * p.grad).abs().detach()
                # Sensitivity smoothing
                self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + (1 - self.beta1) * self.ipt[n]
                # Uncertainty quantification
                self.exp_avg_unc[n] = (
                    self.beta2 * self.exp_avg_unc[n] + (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()
                )

def adalora_peft(model,task_type,rank=8,peft_config=None):
    """
    loss = model(**input).loss
    loss.backward()
    optimizer.step()
    model.base_model.update_and_allocate(i_step)
    optimizer.zero_grad()
    """
    RankAllocator.update_ipt = _update_ipt
    if not peft_config:
        peft_config = AdaLoraConfig(
            peft_type="ADALORA", 
            task_type=task_type, 
            r=rank, 
            lora_alpha=32, 
            lora_dropout=0.01)
    ada_model = get_peft_model(model, peft_config)
    return ada_model
def ia3_peft(model,task_type,peft_config=None):
    if not peft_config:
        peft_config = IA3Config(
            peft_type="IA3",
            task_type=task_type,
            feedforward_modules=["w0"])
    ia3_model = get_peft_model(model, peft_config)
    return ia3_model
def lora_peft(model,task_type,rank=8,peft_config=None):
    if not peft_config:
        peft_config = LoraConfig(
            task_type=task_type,
            r=rank,
            lora_alpha=32,
            lora_dropout=0.01)
    lora_model = get_peft_model(model, peft_config)
    return lora_model
def p_tuning(model,task_type,virtual_tokens=20,token_dim=768,hidden_size=768,att_heads = 12,num_layers=12,peft_config=None):
    if not peft_config:
        peft_config = PromptEncoderConfig(
            peft_type="P_TUNING",
            task_type=task_type,
            num_virtual_tokens=virtual_tokens,
            token_dim=token_dim,
            num_attention_heads=att_heads,
            num_layers=num_layers,
            encoder_hidden_size=hidden_size)
    p_model = get_peft_model(model, peft_config)
    return p_model

def prefix_tuning(model,task_type,virtual_tokens=20,token_dim=768,hidden_size=768,att_heads = 12,num_layers=12,peft_config=None):
    if not peft_config:
        peft_config = PrefixTuningConfig(
            peft_type="PREFIX_TUNING",
            task_type=task_type,
            num_virtual_tokens=virtual_tokens,
            token_dim=token_dim,
            num_transformer_submodules=1,
            num_attention_heads=att_heads,
            num_layers=num_layers,
            encoder_hidden_size=hidden_size)
    prefix_model = get_peft_model(model, peft_config)
    return prefix_model
def prompt_tuning(model,task_type,virtual_tokens=20,token_dim=768,att_heads = 12,num_layers=12,tokenizer_name_or_path=None,prompt_tuning_init_text=None,peft_config=None):
    if not peft_config:
        if prompt_tuning_init_text is None:
            prompt_tuning_init_text = "Predict if sentiment of this review is positive, negative or neutral"
        peft_config = PromptTuningConfig(
            peft_type="PROMPT_TUNING",
            task_type=task_type,
            num_virtual_tokens=virtual_tokens,
            token_dim=token_dim,
            num_transformer_submodules=1,
            num_attention_heads=att_heads,
            num_layers=num_layers,
            prompt_tuning_init="TEXT",
            prompt_tuning_init_text=prompt_tuning_init_text,
            tokenizer_name_or_path=tokenizer_name_or_path)
    prompt_model = get_peft_model(model, peft_config)
    return prompt_model
def get_model_config(config):
    hidden_size,att_heads,num_layers,token_dim = None,None,None,None
    if hasattr(config,"hidden_size"):
        hidden_size = config.hidden_size
    
    if hasattr(config,"n_embd"):
        token_dim = config.n_embd
    if hasattr(config,"word_embed_proj_dim"):
        token_dim = config.word_embed_proj_dim
    
    if hasattr(config,"num_attention_heads"):
        att_heads = config.num_attention_heads
    if hasattr(config,"num_heads"):
        att_heads = config.num_heads
    if hasattr(config,"n_head"):
        att_heads = config.n_head
    
    if hasattr(config,"num_hidden_layers"):
        num_layers  = config.num_hidden_layers
    if hasattr(config,"n_layer"):
        num_layers = config.n_layer
    if hasattr(config,"num_layers"):
        num_layers = config.num_layers
    
    if not token_dim:
        token_dim = hidden_size
    return hidden_size,att_heads,num_layers,token_dim
def peft_function(model,config,peft_type,task_type,rank=8,virtual_tokens=20,tokenizer_name_or_path=None,init_text=None,peft_config=None):
    hidden_size,att_heads,num_layers,token_dim =get_model_config(config)
    if "adalora" == peft_type.lower():
        return adalora_peft(model,task_type,rank=rank,peft_config=None)
    if "lora" == peft_type.lower():
        return lora_peft(model,task_type,rank=rank,peft_config=None)
    if "ia3" == peft_type.lower():
        return ia3_peft(model,task_type,peft_config=None)
    if "p_tuning" == peft_type.lower():
        return p_tuning(model,task_type,
                        virtual_tokens=virtual_tokens,
                        token_dim=token_dim,
                        hidden_size=hidden_size,
                        att_heads = att_heads,
                        num_layers= num_layers,
                        peft_config=None)
    if "prefix_tuning" == peft_type.lower():
        return prefix_tuning(model,task_type,
                        virtual_tokens=virtual_tokens,
                        token_dim=token_dim,
                        hidden_size=hidden_size,
                        att_heads = att_heads,
                        num_layers= num_layers,
                        peft_config=None)
    if "prompt_tuning" == peft_type.lower():
        return prompt_tuning(model,task_type,
                        virtual_tokens=virtual_tokens,
                        token_dim=token_dim,
                        att_heads = att_heads,
                        num_layers=num_layers,
                        tokenizer_name_or_path = tokenizer_name_or_path,
                        prompt_tuning_init_text=init_text,
                        peft_config=None)
    else:
        raise ValueError("unsupported {} peft fine-tuning mode".format(peft_type))
